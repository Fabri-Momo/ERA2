import importlib
import logging

logger = logging.getLogger(__name__)

# Explicit module list: pkgutil.iter_modules is unreliable under PyInstaller.
_MODULES = ('zca', 'pca', 'cholesky', 'fastICA')


def get_processors():
    processors_list = []
    for name in _MODULES:
        full_name = f'{__name__}.{name}'
        try:
            module = importlib.import_module(full_name)
        except Exception:
            # A programming error must not silently remove an algorithm.
            logger.exception('Failed to import processor module %s', full_name)
            continue
        try:
            processor = module.Processor
        except AttributeError:
            logger.debug('Module %s has no Processor class', full_name)
            continue
        processors_list.append(processor)
    if not processors_list:
        raise RuntimeError('No whitening processor could be loaded.')
    return sorted(processors_list, key=lambda x: x.order)
