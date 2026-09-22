import sys
import logging
import numpy
import os
from widgets import *
from imagedata import ImageData, color_interpreters_dict
from qt import QtGui, QtWidgets, QtCore
import qrc_resources
from processors.utils import InsufficientSelectionError
from theme import apply_theme

import cv2
from scipy.spatial import cKDTree
from scipy.stats import ttest_ind
from cleanlab.classification import CleanLearning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage.segmentation import slic
import qimage2ndarray

logger = logging.getLogger(__name__)
XWin = None
YWin = None

DEFAULT_BLUR = 5
DEFAULT_PCA_SIZE = 97
DEFAULT_N_SIZE = 30
DEFAULT_GAIN = 1
DEFAULT_PEN_SIZE = 10
DEFAULT_SUPERPIXEL_COUNT = 50000

# Minimum labelled pixels per class before a classifier is trained.
MIN_SUPERVISION_PIXELS = 20
# Rows per chunk when transforming/predicting a full image.
PREDICT_CHUNK = 500_000

if getattr(sys, 'frozen', False):
    root_folder = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
else:
    root_folder = os.path.dirname(os.path.abspath(__file__))
doc_folder = os.path.join(root_folder, 'resources', 'doc')


def read_version():
    """Version string from the VERSION file (stamped by CI), or 'dev'."""
    try:
        with open(os.path.join(root_folder, 'VERSION'), encoding='utf-8') as f:
            return f.read().strip() or 'dev'
    except OSError:
        return 'dev'

APP_VERSION = read_version()


def _log_dir():
    """Per-user writable directory for the application log."""
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~')
    elif sys.platform == 'darwin':
        base = os.path.expanduser('~/Library/Logs')
    else:
        base = os.environ.get('XDG_STATE_HOME') or os.path.expanduser('~/.local/state')
    return os.path.join(base, 'ERA')


def setup_std_streams():
    """Give the process usable stdout/stderr and a log file.

    In a windowed (console-less) executable sys.stdout and sys.stderr are
    None; any library that writes or flushes them (cleanlab does
    ``sys.stdout.flush()``) would then crash the computation.  Redirect them
    to a per-user log file, which also records uncaught worker errors.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return None
    try:
        log_dir = _log_dir()
        os.makedirs(log_dir, exist_ok=True)
        stream = open(os.path.join(log_dir, 'era.log'), 'a',
                      buffering=1, encoding='utf-8', errors='replace')
    except OSError:
        stream = open(os.devnull, 'w')
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    return stream


def _resolve_about_path():
    """Return the English about.txt (the UI is always in English)."""
    candidates = [
        os.path.join(doc_folder, 'en', 'about.txt'),
        os.path.join(root_folder, 'doc', 'about.txt'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[-1]

left_to_right = True


class _Cancelled(Exception):
    """Raised inside a worker job when cancellation was requested."""


class WorkerThread(QtCore.QThread):
    """Run a heavy job off the GUI thread.

    The job callable receives the worker and may call worker.report(step,
    total) for progress; it must not touch widgets.  Results come back through
    the succeeded signal (executed in the GUI thread).
    """
    progressed = QtCore.Signal(int, int)
    failed = QtCore.Signal(str)
    succeeded = QtCore.Signal(object)

    def __init__(self, job, parent=None):
        super().__init__(parent)
        self._job = job

    def run(self):
        try:
            result = self._job(self)
        except _Cancelled:
            return
        except Exception as exc:
            logger.exception('Worker job failed')
            self.failed.emit(str(exc) or repr(exc))
            return
        if not self.isInterruptionRequested():
            self.succeeded.emit(result)

    def report(self, step, total):
        if self.isInterruptionRequested():
            raise _Cancelled()
        self.progressed.emit(step, total)


def convert_QImage_to_mask(image):
    arr = qimage2ndarray.rgb_view(image)
    height, width, _ = arr.shape
    mask = (arr != 255).any(axis=2)
    if not mask.any():
        return None
    return mask


def save_contours_as_svg(mask, path, width, height):
    """Vectorise the figure in *mask* (figure = 0/black, background = 255).

    cv2.findContours detects *non-zero* regions, so the mask is inverted
    first: the figure becomes the foreground.  Running findContours on the
    raw mask would instead outline the white background and the image frame.
    RETR_TREE keeps internal holes of the figures.
    """
    foreground = 255 - mask
    contours, _ = cv2.findContours(foreground, cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_SIMPLE)
    with open(path, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        for c in contours:
            if len(c) == 0:
                continue
            points = c.reshape(-1, 2)
            d = "M " + " L ".join(f"{int(x)} {int(y)}" for x, y in points) + " Z"
            f.write(f'  <path d="{d}" fill="none" stroke="black" stroke-width="1"/>\n')
        f.write("</svg>\n")


def check_supervision(case_mask, not_case_mask,
                      min_pixels=MIN_SUPERVISION_PIXELS):
    """Validate include/exclude masks before training.

    Returns a human-readable problem, or None when supervision is usable.
    Pixels present in both masks carry contradictory labels: this is an
    error, not something to resolve silently.
    """
    if (case_mask & not_case_mask).any():
        return ('Include and exclude supervision regions overlap.\n'
                'Please correct the supervision before training.')
    if (int(case_mask.sum()) < min_pixels
            or int(not_case_mask.sum()) < min_pixels):
        return 'Supervision regions are too small to train a classifier.'
    return None


# The label-issue filter runs on <= 50k samples: a process pool would cost
# more in spawn overhead (and, frozen, in extra ERA.exe launches) than it
# saves.  seed=0 makes the CV shuffle reproducible run to run.
CLEANLAB_KWARGS = dict(seed=0, find_label_issues_kwargs={'n_jobs': 1})


class FastKNeighborsClassifier(KNeighborsClassifier):
    """KNeighborsClassifier whose predict() uses a multi-threaded cKDTree query.

    Same neighbours, same majority vote, same result as the parent class;
    only the query is parallel (sklearn's tree query is not).  fit and
    predict_proba are inherited unchanged.
    """
    def predict(self, X):
        if getattr(self, 'outputs_2d_', False) or self.weights != 'uniform' \
                or self.metric not in ('minkowski', 'euclidean') or self.p != 2:
            return super().predict(X)
        X = numpy.asarray(X, dtype=numpy.float64)
        _, idx = cKDTree(self._fit_X).query(X, k=self.n_neighbors, workers=-1)
        idx = idx.reshape(len(X), -1)
        votes = self._y[idx]
        counts = numpy.stack([(votes == c).sum(axis=1)
                              for c in range(len(self.classes_))], axis=1)
        return self.classes_[counts.argmax(axis=1)]


def get_classifiers(prediction_method):
    """Return (clf_with_noisy_labels, baseline_clf, extra_scaler)."""
    if prediction_method == 'LR':
        base = LogisticRegression(solver='lbfgs', max_iter=10000)
        clfwn = CleanLearning(clf=LogisticRegression(solver='lbfgs', max_iter=10000),
                              **CLEANLAB_KWARGS)
        return clfwn, base, None

    if prediction_method == 'K-NN':
        base = FastKNeighborsClassifier(n_neighbors=11, n_jobs=-1)
        clfwn = CleanLearning(clf=FastKNeighborsClassifier(n_neighbors=11, n_jobs=-1),
                              **CLEANLAB_KWARGS)
        return clfwn, base, None

    if prediction_method == 'SVM':
        base = LinearSVC(dual='auto', max_iter=10000)
        clfwn = CleanLearning(clf=CalibratedClassifierCV(LinearSVC(dual='auto', max_iter=10000)),
                              **CLEANLAB_KWARGS)
        return clfwn, base, StandardScaler()

    if prediction_method == 'Superpixels':
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
        )
        base = rf
        clfwn = CleanLearning(clf=RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced',
        ), **CLEANLAB_KWARGS)
        return clfwn, base, None

    raise ValueError(f"Unknown prediction method: {prediction_method}")


def process_superpixels(data, case_mask, not_case_mask, blur,
                        n_segments, compactness, worker):
    """Segment with SLIC, train on superpixel means, then broadcast labels."""
    height, width = data.raw_display.shape[:2]

    segments = slic(
        data.raw_display,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        start_label=0,
        channel_axis=-1,
    )
    n_sp = int(segments.max()) + 1
    flat_seg = segments.ravel()

    # Colour features averaged per superpixel, accumulated one
    # colour-space transform at a time (no giant per-pixel matrix).
    n_combos = len(data.processor_names) * len(data.colorspace_names)
    counts = numpy.maximum(
        numpy.bincount(flat_seg, minlength=n_sp), 1)
    superpixel_features = numpy.zeros(
        (n_sp, 3 * n_combos), dtype=numpy.float64)
    total = n_combos + 4
    for i, (proc, cs, native) in enumerate(data.iter_natives()):
        worker.report(i + 1, total)
        blurred = cv2.blur(native, (blur, blur))
        for c in range(3):
            superpixel_features[:, 3 * i + c] = (
                numpy.bincount(flat_seg,
                               weights=blurred[..., c].ravel(),
                               minlength=n_sp)
                / counts
            )

    # Label superpixels from include/exclude masks (majority vote)
    case_counts = numpy.bincount(flat_seg, weights=case_mask.ravel(),
                                 minlength=n_sp)
    not_case_counts = numpy.bincount(flat_seg, weights=not_case_mask.ravel(),
                                     minlength=n_sp)
    labeled = (case_counts + not_case_counts) > 0
    y_sp = numpy.where(case_counts > not_case_counts, 1, 0)

    if not labeled.any() or len(numpy.unique(y_sp[labeled])) < 2:
        raise ValueError(
            "Supervision must cover at least one superpixel for both classes.")

    X_train = superpixel_features[labeled]
    y_train = y_sp[labeled]

    clfwn, clf, _ = get_classifiers('Superpixels')

    worker.report(n_combos + 1, total)
    clfwn.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    worker.report(n_combos + 2, total)

    r = clfwn.predict(superpixel_features)
    r2 = clf.predict(superpixel_features)
    worker.report(total, total)

    # Broadcast superpixel labels to every pixel
    result = r[segments]
    result2 = r2[segments]
    return result.reshape((height, width)), result2.reshape((height, width))


def collect_supervised(data, case_mask, not_case_mask, blur, worker, total):
    """Pass 1: blurred channel values at the supervised pixels only."""
    case_cols, not_cols = [], []
    for i, (proc, cs, native) in enumerate(data.iter_natives()):
        worker.report(i + 1, total)
        blurred = cv2.blur(native, (blur, blur))
        case_cols.append(blurred[case_mask])
        not_cols.append(blurred[not_case_mask])
    return (numpy.concatenate(case_cols, axis=1),
            numpy.concatenate(not_cols, axis=1))


def prepare_train_data(color_case, color_notcase, n_best):
    import warnings
    with warnings.catch_warnings():
        # scipy warns about catastrophic cancellation on (nearly)
        # constant channels; the NaN/Inf scores are handled below.
        warnings.simplefilter('ignore', RuntimeWarning)
        # Vectorized Welch's t-test over all channels at once
        stat, _ = ttest_ind(color_case, color_notcase,
                            equal_var=False, axis=0)
    # Constant channels / tiny classes yield NaN or +-Inf statistics.
    # NaN -> 0 (non-discriminative); +-Inf stays top-ranked.
    stat = numpy.asarray(stat, dtype=numpy.float64)
    big = numpy.finfo(numpy.float64).max
    sel = numpy.abs(numpy.nan_to_num(stat, nan=0.0,
                                     posinf=big, neginf=-big))

    n_best = min(n_best, sel.size)
    best = numpy.argsort(sel)[-n_best:]
    selected = numpy.zeros(sel.size, dtype=bool)
    selected[best] = True

    X_train = numpy.vstack((color_case[:, selected], color_notcase[:, selected]))
    y_train = numpy.asarray(
        [1] * color_case.shape[0] + [0] * color_notcase.shape[0],
        dtype=int,
    )
    return X_train, y_train, selected


def collect_selected(data, selected, blur, worker, total, step0):
    """Pass 2: full-image values of the selected channels only."""
    height, width = data.raw_display.shape[:2]
    n_sel = int(selected.sum())
    cols = numpy.empty((height * width, n_sel), dtype=numpy.float32)
    k = 0
    for i, (proc, cs) in enumerate(
            (p, c) for p in data.processor_names
            for c in data.colorspace_names):
        worker.report(step0 + i + 1, total)
        if not selected[3 * i:3 * i + 3].any():
            continue
        native = color_interpreters_dict[cs].interpret(data.whitened[proc])
        blurred = cv2.blur(native, (blur, blur))
        for c in range(3):
            if selected[3 * i + c]:
                cols[:, k] = blurred[..., c].ravel()
                k += 1
    return cols


def subsample_training_data(X_train, y_train, max_per_class=5000):
    """Randomly cap each class to max_per_class samples to speed up K-NN."""
    rng = numpy.random.default_rng(seed=42)
    indices = numpy.arange(X_train.shape[0])
    sampled = []
    for label in numpy.unique(y_train):
        label_idx = indices[y_train == label]
        if len(label_idx) > max_per_class:
            label_idx = rng.choice(label_idx, max_per_class, replace=False)
        sampled.append(label_idx)
    sampled = numpy.concatenate(sampled)
    return X_train[sampled], y_train[sampled]


def run_classification(data, case_mask, not_case_mask,
                       prediction_method, params, worker):
    """Heavy compute only — worker thread, no widget access."""
    height, width = data.raw_display.shape[:2]
    blur = params['blur']

    if prediction_method == 'Superpixels':
        result, result2 = process_superpixels(
            data, case_mask, not_case_mask, blur,
            params['n_segments'], params['compactness'], worker)
    else:
        n_combos = len(data.processor_names) * len(data.colorspace_names)
        n_chunks = -(-height * width // PREDICT_CHUNK)
        total = 2 * n_combos + n_chunks + 4

        color_case, color_notcase = collect_supervised(
            data, case_mask, not_case_mask, blur, worker, total)

        if (numpy.unique(color_case, axis=0).shape[0] < 2
                or numpy.unique(color_notcase, axis=0).shape[0] < 2):
            raise ValueError(
                'Supervision has insufficient feature variation '
                'to train a classifier.')

        X_train, y_train, selected = prepare_train_data(
            color_case, color_notcase, params['n_best'])

        if prediction_method == 'K-NN':
            X_train, y_train = subsample_training_data(
                X_train, y_train, max_per_class=5000)

        worker.report(n_combos + 1, total)

        scaler = StandardScaler()
        X_train_STD_0 = scaler.fit_transform(X_train)

        pca_size = params['pca_pct']
        variance_exp = 0.9999999 if pca_size == 100 else pca_size / 100

        pca = PCA(variance_exp).fit(X_train_STD_0)
        X_train_STD = pca.transform(X_train_STD_0)

        clfwn, clf, extra_scaler = get_classifiers(prediction_method)
        if extra_scaler is not None:
            X_train_STD = extra_scaler.fit_transform(X_train_STD)

        worker.report(n_combos + 2, total)
        clfwn.fit(X_train_STD, y_train)
        clf.fit(X_train_STD, y_train)
        worker.report(n_combos + 3, total)

        cols = collect_selected(
            data, selected, blur, worker, total, n_combos + 3)

        # Full-image transform + predict, in bounded chunks.
        r = numpy.empty(height * width, dtype=numpy.int64)
        r2 = numpy.empty(height * width, dtype=numpy.int64)
        for j, s in enumerate(range(0, height * width, PREDICT_CHUNK)):
            e = min(s + PREDICT_CHUNK, height * width)
            xc = scaler.transform(cols[s:e])
            xp = pca.transform(xc)
            if extra_scaler is not None:
                xp = extra_scaler.transform(xp)
            r[s:e] = clfwn.predict(xp)
            r2[s:e] = clf.predict(xp)
            worker.report(2 * n_combos + 3 + j + 1, total)

        result = r.reshape((height, width))
        result2 = r2.reshape((height, width))

    # Masks: pixels belonging to the figure become 0 (black), background 255
    mask_clfwn = (result == 0).astype(numpy.uint8) * 255
    mask_clf = (result2 == 0).astype(numpy.uint8) * 255

    # White background: keep original colors inside the figure, white elsewhere
    white_bg = data.raw_display.copy()
    white_bg[result == 0] = [255, 255, 255]

    # Black foreground: black inside the figure, original colors elsewhere
    black_fg = data.raw_display.copy()
    black_fg[result == 1] = [0, 0, 0]

    return {
        'with confident learning': (mask_clfwn, False),
        'without confident learning': (mask_clf, False),
        'white background': (white_bg, True),
        'black foreground': (black_fg, True),
    }


def empty_layout(layout):
    while layout.count():
        child = layout.takeAt(0)
        widget = child.widget()
        if widget is not None:
            widget.setParent(None)




def active_tab_title(tablayout):
    current_index = tablayout.currentIndex()
    if current_index > -1:
        return tablayout.tabText(current_index)
    else:
        return None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.graphics_scene = None
        self._worker = None
        self._busy = False
        self.setWindowTitle(f"ERA — Extraction of Rock Art  v{APP_VERSION}")
        self.setWindowIcon(QtGui.QIcon(':/icon.png'))
        self.data = ImageData()
        self.central_widget = QtWidgets.QWidget()               
        self.setCentralWidget(self.central_widget)
        lay = QtWidgets.QVBoxLayout(self.central_widget)
        label_image = QtWidgets.QLabel(self)
        pixmap0 = QtGui.QPixmap(':/icon.png')
        label_image.setPixmap(pixmap0)
        label_image.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(label_image)
        self.setup_ui()

    def setup_ui(self):

        # actions
        
        open_action = QtWidgets.QAction(QtGui.QIcon(':/open.svg'), self.tr('Open'), self)  
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.setStatusTip(self.tr("Open an image"))
        open_action.triggered.connect(self.load_data)
        
        exit_action = QtWidgets.QAction(QtGui.QIcon(':/exit.svg'), self.tr("Exit"), self)
        exit_action.setShortcut(QtGui.QKeySequence.Quit)
        exit_action.setStatusTip(self.tr("Close application"))
        exit_action.triggered.connect(self.close)

        snap_action = QtWidgets.QAction(QtGui.QIcon(':/snap.svg'), self.tr('Snapshot'), self)  
        snap_action.setShortcut(QtGui.QKeySequence.Save)
        snap_action.setStatusTip(self.tr("Capture the main window"))
        snap_action.triggered.connect(self.snapshot)
        
        
        info_action = QtWidgets.QAction(self.tr('Help'), self)
        info_action.setShortcut(QtGui.QKeySequence.HelpContents)
        info_action.setStatusTip(self.tr("General workflow described"))
        info_action.triggered.connect(self.display_help)
        
        about_action = QtWidgets.QAction(self.tr('About...'), self)  
        about_action.setStatusTip(self.tr("Contributions"))
        about_action.triggered.connect(self.display_about)

        # menu
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu(self.tr("File"))
        self.snap_menu = self.menu.addMenu(self.tr('Capture'))
        self.info_menu = self.menu.addMenu(self.tr('?'))

        self.file_menu.addAction(open_action)
        self.file_menu.addAction(exit_action)

        self.snap_menu.addAction(snap_action)

        self.info_menu.addAction(info_action)
        self.info_menu.addAction(about_action)

        # tool bar

        self.toolbar = self.addToolBar( self.tr("Standard ToolBar") )
        self.toolbar.addAction(open_action)
        self.toolbar.addAction(snap_action)
        self.toolbar.addAction(exit_action)

        # status bar
        self.status = self.statusBar()
        self.status.showMessage(self.tr("Ready to rock"))
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setFixedWidth(260)
        self.progress_bar.setTextVisible(False)
        self.status.addPermanentWidget(self.progress_bar)

        ### layout
        ## global tab structure with two tabs
        ## first tab contains supervision utilities
        ## second tab contains result and refinement parameters

        self.tab_widget = QtWidgets.QTabWidget()
        #self.setCentralWidget(self.tab_widget)


        ## first tab: decorrelation and supervision (sv)
        decorrelation_and_supervision_tab_widget = QtWidgets.QWidget(self.tab_widget)
        self.tab_widget.addTab(decorrelation_and_supervision_tab_widget, self.tr('Supervision'))
        self.tab_widget.setCurrentIndex(0)
        self.tab_widget.currentChanged.connect(self.reset_ml_method)

        decorrelation_and_supervision_tab_layout = QtWidgets.QHBoxLayout(decorrelation_and_supervision_tab_widget)
        decorrelation_and_supervision_tab_splitter = QtWidgets.QSplitter(decorrelation_and_supervision_tab_widget)
        decorrelation_and_supervision_tab_layout.addWidget(decorrelation_and_supervision_tab_splitter)
        #if not left_to_right:
        #    decorrelation_and_supervision_tab_layout.setDirection(QtWidgets.QBoxLayout.RightToLeft)
        decorrelation_and_supervision_tab_widget.setLayout(decorrelation_and_supervision_tab_layout)

        decorrelation_widget = QtWidgets.QWidget(decorrelation_and_supervision_tab_widget)
        self.decorrelation_layout = QtWidgets.QVBoxLayout(decorrelation_widget)
        decorrelation_widget.setLayout(self.decorrelation_layout)
        decorrelation_and_supervision_tab_splitter.addWidget(decorrelation_widget)

        supervision_widget = QtWidgets.QWidget(decorrelation_and_supervision_tab_widget)
        supervision_layout = QtWidgets.QVBoxLayout(supervision_widget)
        supervision_widget.setLayout(supervision_layout)
        decorrelation_and_supervision_tab_splitter.addWidget(supervision_widget)
        decorrelation_and_supervision_tab_splitter.setStretchFactor(2,5)

        # drawing
        self.image_widget = ImageWidget(parent=supervision_widget)
        supervision_layout.addWidget(self.image_widget, stretch=1)


        # channels
        channel_widget = QtWidgets.QWidget(supervision_widget)
        supervision_layout.addWidget(channel_widget, stretch=0)

        outer_channel_layout = QtWidgets.QHBoxLayout()
        self.channel_layout = QtWidgets.QHBoxLayout()
        outer_channel_layout.addStretch()
        inner_channel_widget = QtWidgets.QWidget(supervision_widget)
        inner_channel_widget.setLayout(self.channel_layout)
        outer_channel_layout.addWidget(inner_channel_widget, stretch=0)
        outer_channel_layout.addStretch()
        supervision_tools_widget = QtWidgets.QWidget()
        supervision_tools_layout = QtWidgets.QFormLayout()
        
        

        include_draw_method_radio = QtWidgets.QRadioButton('include')
        exclude_draw_method_radio = QtWidgets.QRadioButton('exclude')
        self.include_pen_size_spinbox = QtWidgets.QSpinBox()
        self.include_pen_size_spinbox.valueChanged.connect(self.set_include_pen_width)
        self.exclude_pen_size_spinbox = QtWidgets.QSpinBox()
        self.exclude_pen_size_spinbox.valueChanged.connect(self.set_exclude_pen_width)
        for spinbox in (self.include_pen_size_spinbox, self.exclude_pen_size_spinbox):
            spinbox.setFixedWidth(80)
        supervision_tools_layout.addRow(include_draw_method_radio, self.include_pen_size_spinbox)
        supervision_tools_layout.addRow(exclude_draw_method_radio, self.exclude_pen_size_spinbox)
        undo_stroke_button = QtWidgets.QPushButton('Undo')
        undo_stroke_button.setToolTip(self.tr("Undo last stroke for current mode"))
        undo_stroke_button.clicked.connect(self.undo_stroke)
        redo_stroke_button = QtWidgets.QPushButton('Redo')
        redo_stroke_button.setToolTip(self.tr("Redo last stroke for current mode"))
        redo_stroke_button.clicked.connect(self.redo_stroke)
        supervision_tools_layout.addRow(undo_stroke_button, redo_stroke_button)
        reset_supervision_button = QtWidgets.QPushButton(self.tr('Reset'))
        reset_supervision_button.setToolTip(self.tr("Reset current mode"))
        reset_supervision_button.clicked.connect(self.reset_strokes)
        resetall_supervision_button = QtWidgets.QPushButton(self.tr('Reset All'))
        resetall_supervision_button.setToolTip(self.tr("Reset all modes"))
        resetall_supervision_button.clicked.connect(self.reset_all_strokes)
        continue_push_button = QtWidgets.QPushButton(self.tr('Continue'))
        continue_push_button.setProperty('primary', True)
        continue_push_button.clicked.connect(self.process)
        supervision_tools_layout.addRow(reset_supervision_button, resetall_supervision_button)
        supervision_tools_layout.addRow(continue_push_button)
        supervision_tools_widget.setLayout(supervision_tools_layout)
        outer_channel_layout.addWidget(supervision_tools_widget, stretch=0)
        channel_widget.setLayout(outer_channel_layout)

        # colorspaces
        #colorspace_container_widget = QtWidgets.QWidget()
        #self.colorspace_container_layout = QtWidgets.QVBoxLayout()
        #colorspace_container_widget.setLayout(self.colorspace_container_layout)
        self.colorspace_container_widget = QtWidgets.QTabWidget()
        self.colorspace_container_widget.setTabPosition(QtWidgets.QTabWidget.West)
        self.colorspace_container_widget.currentChanged.connect(self.reparenting_ref_colorbundle)


        self.decorrelation_layout.addWidget(self.colorspace_container_widget, stretch=2)
        
        ## decorrelation
        self.process_widget = QtWidgets.QWidget(decorrelation_widget)

        process_form_widget = QtWidgets.QWidget()
        process_form_layout = QtWidgets.QFormLayout()
        process_form_widget.setLayout(process_form_layout)
        self.decorrelation_layout.addWidget(self.process_widget, stretch=0)

        select_draw_method_radio = QtWidgets.QRadioButton('select')
        self.select_pen_size_spinbox = QtWidgets.QSpinBox()
        self.select_pen_size_spinbox.valueChanged.connect(self.set_select_pen_width)
        process_form_layout.addRow(self.tr('Stroke Width'), self.select_pen_size_spinbox)
        self.contrast_boost_spinbox = QtWidgets.QSpinBox()
        self.contrast_boost_spinbox.setRange(0, 20)
        process_form_layout.addRow(self.tr('Contrast Boost'), self.contrast_boost_spinbox)
        process_push_button = QtWidgets.QPushButton(self.tr('Decorrelation Refinement'))
        process_push_button.setProperty('primary', True)
        process_push_button.clicked.connect(self.decorrelate)


        process_box = QtWidgets.QVBoxLayout()
        process_box.addWidget(select_draw_method_radio)
        process_box.addWidget(process_form_widget)
        process_box.addWidget(process_push_button)

        self.process_widget.setLayout(process_box)

        ## second tab: result and refinement (rr)
        drawing_tab_widget = QtWidgets.QWidget(self.tab_widget)
        self.tab_widget.addTab(drawing_tab_widget, self.tr('Drawing'))
        self.tab_widget.setTabEnabled(1, False)

        drawing_tab_layout = QtWidgets.QHBoxLayout(drawing_tab_widget)
        drawing_tab_widget.setLayout(drawing_tab_layout)

        # left part
        drawing_widget = QtWidgets.QWidget(drawing_tab_widget)
        drawing_tab_layout.addWidget(drawing_widget)
        drawing_grid = QtWidgets.QGridLayout(drawing_widget)
        drawing_widget.setLayout(drawing_grid)

        self.results = {'with confident learning': {},
                        'without confident learning': {},
                        'white background': {},
                        'black foreground': {},
                        }
        self.results['with confident learning'].setdefault('view', ResultWidget(label='With confident learning'))
        self.results['without confident learning'].setdefault('view', ResultWidget(label='Without confident learning'))
        self.results['white background'].setdefault('view', ResultWidget(label='White background'))
        self.results['black foreground'].setdefault('view', ResultWidget(label='Black foreground'))

        result_coords = [(i//2, i%2) for i in range(len(self.results))]
        for result, coords in zip(self.results.values(), result_coords):
            drawing_grid.addWidget(result['view'], *coords)

        # right part
        refine_widget = QtWidgets.QWidget(drawing_tab_widget)
        refine_layout = QtWidgets.QFormLayout()
        refine_widget.setLayout(refine_layout)

        drawing_tab_layout.addWidget(refine_widget)
        #refine_layout = QtWidgets.QVBoxLayout(refine_widget)
        refine_widget.setLayout(refine_layout)
        self.blursize = QtWidgets.QSpinBox(self)
        self.blursize.setSingleStep(1)
        self.blursize.setRange(1,50)
        self.blursize.setValue(DEFAULT_BLUR)
        refine_layout.addRow(self.tr('Blur Radius'), self.blursize)

        self.PCAsize = QtWidgets.QSpinBox(self)
        self.PCAsize.setSingleStep(1)
        self.PCAsize.setRange(75,100)
        self.PCAsize.setValue(DEFAULT_PCA_SIZE)
        refine_layout.addRow(self.tr('PCA var. explained:'), self.PCAsize)
        #self.PCAsize.valueChanged.connect(self.valuechange4)

        self.Nsize = QtWidgets.QSpinBox(self)
        self.Nsize.setSingleStep(1)
        self.Nsize.setRange(10,102)
        self.Nsize.setValue(DEFAULT_N_SIZE)
        refine_layout.addRow(self.tr('n best channels:'), self.Nsize)
        #self.Nsize.valueChanged.connect(self.valuechange5)
        #
        self.prediction_button_lr = QtWidgets.QRadioButton('LR')
        prediction_button_svm = QtWidgets.QRadioButton('SVM')
        prediction_button_knn = QtWidgets.QRadioButton('K-NN')
        prediction_button_superpixels = QtWidgets.QRadioButton('Superpixels')
        self.prediction_button_lr.setChecked(True)

        refine_layout.addRow('Prediction Method:', self.prediction_button_lr)
        refine_layout.addRow('', prediction_button_svm)
        refine_layout.addRow('', prediction_button_knn)
        refine_layout.addRow('', prediction_button_superpixels)

        self.superpixel_count_spinbox = QtWidgets.QSpinBox(self)
        self.superpixel_count_spinbox.setRange(50, 100000)
        self.superpixel_count_spinbox.setSingleStep(50)
        self.superpixel_count_spinbox.setValue(DEFAULT_SUPERPIXEL_COUNT)
        refine_layout.addRow(self.tr('Superpixel count:'), self.superpixel_count_spinbox)

        self.superpixel_compactness_spinbox = QtWidgets.QSpinBox(self)
        self.superpixel_compactness_spinbox.setRange(1, 100)
        self.superpixel_compactness_spinbox.setValue(10)
        refine_layout.addRow(self.tr('Superpixel compactness:'), self.superpixel_compactness_spinbox)

        btnReprocess = QtWidgets.QPushButton(self.tr("Re-process"), self)
        btnReprocess.setProperty('primary', True)
        btnReprocess.setMinimumWidth(110)
        btnReprocess.setToolTip(self.tr("Reprocess <i>Calculate</i>"))      
        btnReprocess.clicked.connect(self.process)
        #
        btnSave = QtWidgets.QPushButton(self.tr("Save"), self)
        btnSave.setMinimumWidth(110)
        btnSave.setToolTip(self.tr("Save the checked images"))  
        refine_layout.addRow(btnReprocess, btnSave)
        btnSave.clicked.connect(self.saveImage)
        #       

        ### selection groups
        self.draw_mode_selection_group = QtWidgets.QButtonGroup()
        self.draw_mode_selection_group.buttonClicked.connect(self.set_pen_mode)
        self.draw_mode_selection_group.addButton(select_draw_method_radio)
        self.draw_mode_selection_group.addButton(include_draw_method_radio)
        self.draw_mode_selection_group.addButton(exclude_draw_method_radio)
        self.channel_selection_group = QtWidgets.QButtonGroup(channel_widget)
        self.channel_selection_group.buttonClicked.connect(self.refresh_canvas)
        self.colorspace_selection_group = QtWidgets.QButtonGroup()
        self.colorspace_selection_group.buttonClicked.connect(self.refresh_channels)
        self.prediction_method_group = QtWidgets.QButtonGroup()
        self.prediction_method_group.addButton(self.prediction_button_lr)
        self.prediction_method_group.addButton(prediction_button_svm)
        self.prediction_method_group.addButton(prediction_button_knn)
        self.prediction_method_group.addButton(prediction_button_superpixels)
        self.current_colorspace = None


        ## second tab

    # ---------------------------------------------------------- worker glue
    def _start_worker(self, job, on_success):
        """Start a background job; only one heavy job runs at a time."""
        old = self._worker
        try:
            old_running = old is not None and old.isRunning()
        except RuntimeError:
            # The C++ object was already deleted via deleteLater.
            old_running = False
        if old_running:
            old.requestInterruption()
            old.wait(5000)
            # The stale worker must not overwrite newer results.
            for signal in (old.succeeded, old.failed, old.progressed):
                try:
                    signal.disconnect()
                except (TypeError, RuntimeError):
                    pass
        worker = WorkerThread(job, self)
        worker.progressed.connect(self._on_worker_progress)
        worker.failed.connect(self._on_worker_failure)
        worker.succeeded.connect(on_success)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        self._set_busy(True)
        worker.start()

    def _on_worker_finished(self, worker):
        # A stale worker finishing late must not re-enable the UI while a
        # newer job is still running.
        if worker is self._worker:
            self._set_busy(False)

    def _set_busy(self, busy):
        """Disable the UI while a heavy job runs (idempotent)."""
        if busy == self._busy:
            return
        self._busy = busy
        self.tab_widget.setEnabled(not busy)
        self.toolbar.setEnabled(not busy)
        if busy:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_worker_progress(self, step, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(step)

    def _on_worker_failure(self, message):
        self._set_busy(False)
        self.progress_bar.reset()
        self.status.clearMessage()
        self.modal_error(message)

    def reset_ml_method(self):
        self.prediction_button_lr.setChecked(True)

    def reset_prediction_parameters(self):
        self.prediction_button_lr.setChecked(True)
        self.Nsize.setValue(DEFAULT_N_SIZE)
        self.PCAsize.setValue(DEFAULT_PCA_SIZE)
        self.blursize.setValue(DEFAULT_BLUR)
        self.superpixel_count_spinbox.setValue(DEFAULT_SUPERPIXEL_COUNT)
        self.superpixel_compactness_spinbox.setValue(10)
        for result in self.results.values():
            view = result['view']
            if view.background_pixmap is not None and view.background_pixmap.scene() is view.scene:
                view.scene.removeItem(view.background_pixmap)
            view.background_pixmap = None
            view.uncheck()
        self.tab_widget.setTabEnabled(1, False)
        self.tab_widget.setCurrentIndex(0)

    def reset_supervision_parameters(self):
        self.draw_mode_selection_group.buttons()[0].click()
        if self.colorspace_selection_group.buttons():
            self.colorspace_selection_group.buttons()[0].click()
        self.set_pen_mode()
        self.select_pen_size_spinbox.setValue(DEFAULT_PEN_SIZE)
        self.include_pen_size_spinbox.setValue(DEFAULT_PEN_SIZE)
        self.exclude_pen_size_spinbox.setValue(DEFAULT_PEN_SIZE)
        for mode in ['select', 'include', 'exclude']:
            self.set_pen_width(mode, DEFAULT_PEN_SIZE)
        self.contrast_boost_spinbox.setValue(DEFAULT_GAIN)

    def _set_result_image(self, key, data, is_color=True):
        """Store data and set its pixmap in the corresponding result view."""
        view = self.results[key]['view']
        self.results[key]['data'] = data

        height, width = data.shape[:2]
        if is_color:
            qimg = QtGui.QImage(data, width, height, 3 * width, QtGui.QImage.Format_RGB888)
        else:
            qimg = QtGui.QImage(data, width, height, width, QtGui.QImage.Format_Grayscale8)
        view.set_background_pixmap(QtGui.QPixmap.fromImage(qimg))

    def process(self):
        if self.graphics_scene is None or self.data.raw_display is None:
            return
        case_mask = convert_QImage_to_mask(self.graphics_scene.render_layer('include'))
        not_case_mask = convert_QImage_to_mask(self.graphics_scene.render_layer('exclude'))
        if case_mask is None or not_case_mask is None:
            message = self.tr("""You must supervise first!

Choose the size of the pen and select the option include / exclude
for painting the areas corresponding to the figure and the surrounding, respectively.""")
            self.modal_error(message)
            return

        problem = check_supervision(case_mask, not_case_mask)
        if problem is not None:
            self.modal_error(self.tr(problem))
            return

        prediction_method = self.prediction_method_group.checkedButton().text()
        self.status.showMessage(self.tr('Running prediction with {} method').format(prediction_method))
        params = dict(
            blur=self.blursize.value(),
            n_best=self.Nsize.value(),
            pca_pct=self.PCAsize.value(),
            n_segments=self.superpixel_count_spinbox.value(),
            compactness=self.superpixel_compactness_spinbox.value(),
        )

        def job(worker):
            return run_classification(
                self.data, case_mask, not_case_mask,
                prediction_method, params, worker)

        self._start_worker(job, self._on_classification_done)

    def _on_classification_done(self, results):
        # Re-enable the UI first: QAbstractButton.click() and tab switching
        # are ignored on disabled widgets.
        self._set_busy(False)
        for key, (data, is_color) in results.items():
            self._set_result_image(key, data, is_color=is_color)
        self.progress_bar.reset()
        self.status.clearMessage()
        self.tab_widget.setTabEnabled(1, True)
        self.tab_widget.setCurrentIndex(1)

    def _save_contours_as_svg(self, mask, path, width, height):
        save_contours_as_svg(mask, path, width, height)

    def saveImage(self):
        results_to_save = [key for key in self.results if self.results[key]['view'].is_selected_for_export()]
        if not results_to_save:
            self.modal_error(self.tr('Please select image to save first'))
            return

        image_name = os.path.splitext(os.path.basename(self.data.image_path))[0]
        image_directory = os.path.dirname(self.data.image_path)
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr('Choose destination folder'),
            image_directory,
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if not directory:
            return

        height, width = self.data.raw_display.shape[:2]

        for key in results_to_save:
            result = self.results[key]
            image_path = os.path.join(directory, f"{image_name}-{key.replace(' ', '_')}.tif")
            data = result['data']

            if data.ndim == 3 and data.shape[2] == 3:
                # data is RGB; cv2.imwrite expects BGR
                cv2.imwrite(image_path, cv2.cvtColor(data, cv2.COLOR_RGB2BGR))
            else:
                # Grayscale mask: already in the right channel layout
                cv2.imwrite(image_path, data)

            if key in ['without confident learning', 'with confident learning']:
                svg_path = os.path.join(directory, f"{image_name}-{key.replace(' ', '_')}.svg")
                self._save_contours_as_svg(result['data'], svg_path, width, height)


    def _current_color_bundle(self):
        """Return the ColorSpaceBundle currently selected by the user."""
        colorspace_button = self.colorspace_selection_group.checkedButton()
        if colorspace_button is None or colorspace_button.text() == 'Original':
            return self.data.reference

        processor = active_tab_title(self.colorspace_container_widget)
        colorspace = colorspace_button.text()
        return self.data.get_bundle(processor, colorspace)

    def _current_channel_image(self, color_bundle):
        """Return the QImage of the currently selected channel/composite."""
        channel_button = self.channel_selection_group.checkedButton()
        if channel_button is None:
            return color_bundle.composite

        label = channel_button.text()
        if label.startswith('Channel '):
            return color_bundle.channels[int(label.split()[-1]) - 1]
        return color_bundle.composite

    def decorrelate(self):
        if self.graphics_scene is None or self.data.raw_display is None:
            return
        image = self.graphics_scene.render_layer('select')
        mask = convert_QImage_to_mask(image)
        contrast_boost = self.contrast_boost_spinbox.value()

        def job(worker):
            self.data.update(mask=mask, contrast_boost=contrast_boost,
                             progress_cb=worker.report,
                             cancel_cb=worker.isInterruptionRequested)

        self._start_worker(job, self._on_decorrelate_done)

    def _on_decorrelate_done(self, _result):
        self._set_busy(False)
        self.refresh_tabbed_colorspace_picker()
        self.progress_bar.reset()
        self.status.clearMessage()

    def reparenting_ref_colorbundle(self):
        current_scroll = self.colorspace_container_widget.currentWidget()
        if not current_scroll:
            return
        colorspace_widget = current_scroll.widget()
        if not colorspace_widget:
            return
        # Reselect the current colorspace in the new tab; fallback to 'Original'
        for item in self.colorspace_selection_group.buttons():
            if item.parentWidget() is colorspace_widget and item.text() == self.current_colorspace:
                item.click()
                return
        for item in self.colorspace_selection_group.buttons():
            if item.parentWidget() is colorspace_widget and item.text() == 'Original':
                item.click()
                return

    def undo_stroke(self):
        if self.graphics_scene is None:
            return
        self.image_widget.history_event('undo')

    def redo_stroke(self):
        if self.graphics_scene is None:
            return
        self.image_widget.history_event('redo')

    def reset_strokes(self):
        if self.graphics_scene is None:
            return
        self.graphics_scene.flush_history(length=0)

    def reset_all_strokes(self):
        if self.graphics_scene is None:
            return
        self.graphics_scene.flush_all_history()

    def set_pen_mode(self):
        if self.graphics_scene is None:
            return
        checked = self.draw_mode_selection_group.checkedButton()
        if checked is None:
            return
        self.graphics_scene.set_pen_mode(checked.text())

    def set_select_pen_width(self):
        size = self.select_pen_size_spinbox.value()
        self.set_pen_width('select', size)

    def set_include_pen_width(self):
        size = self.include_pen_size_spinbox.value()
        self.set_pen_width('include', size)

    def set_exclude_pen_width(self):
        size = self.exclude_pen_size_spinbox.value()
        self.set_pen_width('exclude', size)

    def set_pen_width(self, mode, size):
        if self.graphics_scene is None:
            return
        self.graphics_scene.set_pen_width(mode, size)

    def refresh_channels(self):
        button = self.colorspace_selection_group.checkedButton()
        if button is None:
            return
        self.current_colorspace = button.text()
        # clean channels list
        empty_layout(self.channel_layout)
        color_bundle = self._current_color_bundle()
        if color_bundle is None:
            return
        # populate group with selected channels
        composite = ColorVariant(label='Composite', thumbnail=color_bundle.composite)
        self.channel_layout.addWidget(composite)
        self.channel_selection_group.addButton(composite)
        composite.click()
        for index, channel in enumerate(color_bundle.channels):
            item = ColorVariant(label=f'Channel {index + 1}', thumbnail=color_bundle.channels[index])
            self.channel_layout.addWidget(item)
            self.channel_selection_group.addButton(item)

    def snapshot(self):
        color_bundle = self._current_color_bundle()
        if color_bundle is None or self.data.image_path is None:
            return
        image = self._current_channel_image(color_bundle)
        image_name, _ = os.path.splitext(os.path.basename(self.data.image_path))
        channel_button = self.channel_selection_group.checkedButton()
        channel_label = channel_button.text() if channel_button else 'Composite'
        suffix = f'{color_bundle.code}-{channel_label}'
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr('Choose destination folder'),
            os.path.dirname(self.data.image_path),
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if not directory:
            return
        image.save(os.path.join(directory, f'{image_name}-{suffix}.tif'))

    def refresh_canvas(self):
        if self.graphics_scene is None:
            return
        color_bundle = self._current_color_bundle()
        if color_bundle is None:
            return
        image = self._current_channel_image(color_bundle)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.graphics_scene.set_background_pixmap(pixmap)

    def refresh_tabbed_colorspace_picker(self):
        current_page = self.colorspace_container_widget.currentIndex()
        if current_page != -1:
            tab_text = self.colorspace_container_widget.tabText(current_page)
        else:
            tab_text = None
        current_colorspace = self.colorspace_selection_group.checkedButton()
        if current_colorspace:
            colorspace_text = current_colorspace.text()
        else:
            colorspace_text = 'Original'
        self.colorspace_container_widget.clear()
        for button in self.colorspace_selection_group.buttons():
            self.colorspace_selection_group.removeButton(button)
        active_colorspace = None
        for processor in self.data.whitened:
            colorspace_scroller = QtWidgets.QScrollArea()
            colorspace_widget = QtWidgets.QWidget(colorspace_scroller)
            colorspace_layout = QtWidgets.QGridLayout(colorspace_widget)
            colorspace_widget.setLayout(colorspace_layout)

            # One 'Original' thumbnail per tab, same size as the others
            original_item = ColorVariant(label='Original', thumbnail=self.data.get_reference_thumbnail())
            colorspace_layout.addWidget(original_item, 0, 0)
            self.colorspace_selection_group.addButton(original_item)
            if (processor == tab_text or tab_text is None) and colorspace_text == 'Original':
                active_colorspace = original_item

            # 2-column grid; 'Original' is at (0, 0), colorspaces fill from (0, 1)
            for index, colorspace in enumerate(self.data.colorspace_names, start=1):
                coords = (index // 2, index % 2)
                item = ColorVariant(label=colorspace, thumbnail=self.data.get_thumbnail(processor, colorspace))
                colorspace_layout.addWidget(item, *coords)
                self.colorspace_selection_group.addButton(item)
                if processor == tab_text or tab_text is None:
                    if colorspace == self.current_colorspace:
                        active_colorspace = item
            colorspace_scroller.setWidget(colorspace_widget)
            colorspace_scroller.setWidgetResizable(True)
            self.colorspace_container_widget.addTab(colorspace_scroller, processor)
        buttons = self.colorspace_selection_group.buttons()
        if current_page != -1:
            self.colorspace_container_widget.setCurrentIndex(current_page)
        if active_colorspace is None and buttons:
            active_colorspace = buttons[0]
        if active_colorspace is not None:
            active_colorspace.click()

    def load_data(self):
        name = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr('Open Image'),
            filter=self.tr("Image Files (*.png *.jpg *.bmp *.tif *.tiff)")
        )
        if not name[0]:
            return

        self.setCentralWidget(self.tab_widget)
        self.reset_prediction_parameters()

        if self.graphics_scene is not None:
            old_scene = self.graphics_scene
            self.image_widget.setScene(None)
            old_scene.deleteLater()
            self.graphics_scene = None

        self.graphics_scene = CustomScene()
        self.image_widget.setScene(self.graphics_scene)
        self.graphics_scene.view = self.image_widget
        self.reset_supervision_parameters()

        self.status.showMessage(self.tr('Loading picture…'))
        image_path = name[0]

        def job(worker):
            self.data.load(image_path,
                           progress_cb=worker.report,
                           cancel_cb=worker.isInterruptionRequested)

        self._start_worker(job, self._on_load_done)

    def _on_load_done(self, _result):
        self._set_busy(False)
        self.refresh_tabbed_colorspace_picker()
        self.progress_bar.reset()
        self.status.clearMessage()
        self.image_widget.init_view()

    def display_help(self):

        form = Help('index.html', self)
        form.show()
    
    def display_about(self):
        about_path = _resolve_about_path()
        with open(about_path, 'r', encoding='utf-8') as about_file:
            message = about_file.read()

        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("About ERA")
        box.setIconPixmap(QtGui.QPixmap(':/icon.png').scaled(
            96, 96, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        box.setText(f"<b>ERA — Extraction of Rock Art</b><br>Version {APP_VERSION}")
        box.setInformativeText(message)
        box.exec()
    
    def modal_error(self, message):
        QtWidgets.QMessageBox.critical(self, "ERA - Error", message)
                

if __name__ == '__main__':
    # Required for multiprocessing (cleanlab) in a frozen executable.
    import multiprocessing
    multiprocessing.freeze_support()
    setup_std_streams()
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logger.info('ERA %s starting', APP_VERSION)
    app = QtWidgets.QApplication(sys.argv)
    # The UI is always in English: no translator is installed, and Qt's own
    # dialogs (file chooser, message boxes) are forced to the C locale.
    QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.C))
    apply_theme(app)
    window = MainWindow()
    window.showMaximized()
    app.exec()
