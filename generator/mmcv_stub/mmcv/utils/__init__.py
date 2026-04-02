"""mmcv.utils stub: re-export equivalents from mmengine."""
try:
    from mmengine.config import Config, DictAction
except ImportError:
    class Config(dict):
        pass
    class DictAction:
        pass

def collect_env():
    """Stub for mmcv.utils.collect_env (not needed for inference)."""
    return {}

def get_git_hash(fallback="unknown", digits=None):
    """Stub for mmcv.utils.get_git_hash (not needed for inference)."""
    return fallback
