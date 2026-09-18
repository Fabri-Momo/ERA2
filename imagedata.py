import logging
from collections import OrderedDict

import cv2
import numpy as np

from qt import QtGui
from color_interpreters import get_color_interpreters
from processors import get_processors
from processors.utils import matrix_stats

logger = logging.getLogger(__name__)

color_interpreters_dict = {ci.name: ci() for ci in get_color_interpreters()}
processors_dict = {proc.name: proc() for proc in get_processors()}

# Number of full-resolution display results kept in memory at once.  Colour
# conversions are recomputed on demand; the cache only limits recomputation.
BUNDLE_CACHE_SIZE = 6

# Maximum side of the thumbnails used in the colour-space picker.  Thumbnails
# never require a full-resolution copy of a result.
THUMBNAIL_MAX_DIM = 320


def load_image(image_path):
    """Read an image preserving its original bit depth.

    Returns (rgb, valid_mask):
      rgb:        (H, W, 3) array in RGB order, original dtype (uint8/uint16/
                  float) — never quantised here.
      valid_mask: (H, W) bool — False where alpha == 0 or values are
                  non-finite.  Pixels are never considered invalid just for
                  being black or white.
    """
    raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise IOError(f'Unable to read image: {image_path}')

    alpha = None
    if raw.ndim == 2:
        rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    elif raw.ndim == 3 and raw.shape[2] == 2:
        gray, alpha = raw[..., 0], raw[..., 1]
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif raw.ndim == 3 and raw.shape[2] == 3:
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    elif raw.ndim == 3 and raw.shape[2] >= 4:
        rgb = cv2.cvtColor(raw[..., :4], cv2.COLOR_BGRA2RGB)
        alpha = raw[..., 3]
    else:
        raise IOError(f'Unsupported image layout {raw.shape}: {image_path}')

    valid_mask = np.ones(rgb.shape[:2], dtype=bool)
    if alpha is not None:
        valid_mask &= alpha > 0
    if np.issubdtype(rgb.dtype, np.floating):
        valid_mask &= np.isfinite(rgb).all(axis=2)
    return rgb, valid_mask


def source_to_unit_float(rgb):
    """Scientific pipeline input: float32 in [0, 1] preserving full bit depth.

    uint8 -> /255, uint16 -> /65535 (keeps all sub-8-bit levels), float data
    is clipped to [0, 1] (scaled by 255 or 65535 if clearly stored on those
    scales).  NaN/Inf become 0.
    """
    if rgb.dtype == np.uint8:
        out = rgb.astype(np.float32) / 255.0
    elif rgb.dtype == np.uint16:
        out = rgb.astype(np.float32) / 65535.0
    elif np.issubdtype(rgb.dtype, np.floating):
        out = rgb.astype(np.float32)
        peak = float(out.max()) if out.size else 0.0
        if peak > 1.0:
            out = out / (255.0 if peak <= 255.0 else 65535.0)
    else:
        out = rgb.astype(np.float32) / float(np.iinfo(rgb.dtype).max)
    return np.nan_to_num(np.clip(out, 0.0, 1.0))


def to_display_uint8(rgb):
    """Display image: uint8 RGB, preserving as much dynamic range as fits."""
    return np.clip(np.round(source_to_unit_float(rgb) * 255.0),
                   0, 255).astype(np.uint8)


def _qimage_rgb(display):
    """QImage (RGB888) viewing a uint8 (H, W, 3) array; keeps the buffer alive."""
    buf = np.ascontiguousarray(display)
    h, w = buf.shape[:2]
    image = QtGui.QImage(buf, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    image._era_buffer = buf  # QImage does not copy: keep the array alive.
    return image


def _qimage_gray(channel):
    buf = np.ascontiguousarray(channel)
    h, w = buf.shape[:2]
    image = QtGui.QImage(buf, w, h, w, QtGui.QImage.Format_Grayscale8)
    image._era_buffer = buf
    return image


class ColorSpaceBundle:
    """Display-ready view of one colour-space transform.

    display: uint8 (H, W, 3) image encoded for Qt (the only data guaranteed
             to be kept).
    native:  optional float32 native colour-space values (unquantised); may be
             None on cached bundles to bound memory usage.
    """
    def __init__(self, code, display, native=None):
        self.code = code
        self.display = np.ascontiguousarray(display)
        self.buffer = self.display  # backwards compatibility
        self.native = native
        self._composite = None
        self._channels = None

    @property
    def composite(self):
        if self._composite is None:
            self._composite = _qimage_rgb(self.display)
        return self._composite

    @property
    def channels(self):
        """Per-channel QImages, created on first access only."""
        if self._channels is None:
            self._channels = [
                _qimage_gray(c) for c in cv2.split(self.display)]
        return self._channels

    def get_blurred(self, blur=5):
        src = self.native if self.native is not None else self.display
        return cv2.blur(src, (blur, blur))

    def get_stats(self):
        src = self.native if self.native is not None else self.display
        print(matrix_stats(src))


class ImageData:
    def __init__(self):
        self.reset()

    def reset(self):
        self.raw_source = None       # original dtype, RGB order
        self.raw_display = None      # uint8 RGB for Qt
        self.valid_mask = None       # bool (H, W)
        self.source_float = None     # float32 (H, W, 3) in [0, 1]
        self.reference = None        # ColorSpaceBundle of the original image
        self.whitened = {}           # processor name -> float32 pseudo-RGB [0,1]
        self._bundles = OrderedDict()  # (processor, colorspace) -> bundle (LRU)
        self.image_path = None

    @property
    def processor_names(self):
        return list(processors_dict)

    @property
    def colorspace_names(self):
        return list(color_interpreters_dict)

    # ------------------------------------------------------------------ load
    def load(self, image_path, progress_cb=None, cancel_cb=None):
        self.reset()
        self.image_path = image_path
        self.raw_source, self.valid_mask = load_image(image_path)
        self.raw_display = to_display_uint8(self.raw_source)
        self.source_float = source_to_unit_float(self.raw_source)
        self.reference = ColorSpaceBundle(
            'RGB', display=self.raw_display, native=self.source_float)
        self._compute_whitened(mask=None, contrast_boost=0,
                               progress_cb=progress_cb, cancel_cb=cancel_cb)

    def update(self, mask=None, contrast_boost=0,
               progress_cb=None, cancel_cb=None):
        self._compute_whitened(mask=mask, contrast_boost=contrast_boost,
                               progress_cb=progress_cb, cancel_cb=cancel_cb)

    def _compute_whitened(self, mask, contrast_boost,
                          progress_cb=None, cancel_cb=None):
        """Run the four whitening transforms on the scientific (float) data.

        Results are float32 pseudo-RGB in [0, 1]; colour-space conversion and
        any 8-bit quantisation happen later, on demand, for display only.
        """
        whitened = {}
        total = len(processors_dict)
        for step, name in enumerate(processors_dict, start=1):
            if cancel_cb is not None and cancel_cb():
                raise InterruptedError('processing cancelled')
            whitened[name] = processors_dict[name].process(
                self.source_float,
                mask=mask,
                contrast_boost=contrast_boost,
                valid_mask=self.valid_mask,
            )
            if progress_cb is not None:
                progress_cb(step, total)
        self.whitened = whitened
        self._bundles.clear()

    # ------------------------------------------------------- colour bundles
    def get_native(self, processor, colorspace):
        """Native (unquantised, float32) colour-space data for a result."""
        return color_interpreters_dict[colorspace].interpret(
            self.whitened[processor])

    def get_bundle(self, processor, colorspace):
        """Full-resolution ColorSpaceBundle, from a small LRU cache."""
        key = (processor, colorspace)
        bundle = self._bundles.get(key)
        if bundle is not None:
            self._bundles.move_to_end(key)
            return bundle
        interpreter = color_interpreters_dict[colorspace]
        native = interpreter.interpret(self.whitened[processor])
        display = interpreter.to_display(native)
        # Native data is dropped: it can be recomputed on demand and keeping
        # 36 full-resolution float images is what used most of the memory.
        bundle = ColorSpaceBundle(colorspace, display=display)
        self._bundles[key] = bundle
        while len(self._bundles) > BUNDLE_CACHE_SIZE:
            self._bundles.popitem(last=False)
        return bundle

    def iter_natives(self):
        """Yield (processor, colorspace, native float32) in historical order.

        Colour conversions are computed on the fly and not retained: callers
        that need every combination (classification) pay one full-resolution
        float image at a time instead of 36.
        """
        for processor in processors_dict:
            whitened = self.whitened[processor]
            for colorspace, interpreter in color_interpreters_dict.items():
                yield processor, colorspace, interpreter.interpret(whitened)

    # ------------------------------------------------------------- thumbnails
    def _downscaled(self, image, max_dim):
        h, w = image.shape[:2]
        scale = max_dim / max(h, w)
        if scale >= 1.0:
            return image
        return cv2.resize(image, (int(round(w * scale)),
                                  int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)

    def get_thumbnail(self, processor, colorspace, max_dim=THUMBNAIL_MAX_DIM):
        """Small QImage for the colour-space picker (no full-res copy)."""
        small = self._downscaled(self.whitened[processor], max_dim)
        interpreter = color_interpreters_dict[colorspace]
        display = interpreter.to_display(interpreter.interpret(small))
        return _qimage_rgb(display)

    def get_reference_thumbnail(self, max_dim=THUMBNAIL_MAX_DIM):
        return _qimage_rgb(self._downscaled(self.raw_display, max_dim))

    # ------------------------------------------------------------- leftovers
    def get_bundles(self, with_reference=True):
        """Compatibility helper: all currently cached bundles + reference."""
        bundles = list(self._bundles.values())
        if with_reference and self.reference is not None:
            bundles.insert(0, self.reference)
        return bundles
