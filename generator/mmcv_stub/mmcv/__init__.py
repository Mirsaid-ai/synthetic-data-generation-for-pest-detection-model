"""Minimal mmcv stub — delegates to mmengine (the official successor).

mmcv cannot be pip-installed on PyTorch 2.7+CUDA12.6 because OpenMMLab has
not released pre-built wheels for this combination yet, and building from
source fails on Python 3.12 due to a pkg_resources removal.

Metric3D v2 inference only uses mmcv.utils.{Config, DictAction,
collect_env, get_git_hash}. All four are provided here via mmengine,
which IS pip-installable and is the official successor to mmcv.

Install: the setup scripts add this directory to PYTHONPATH automatically.
"""
from mmcv import utils  # noqa: F401 — makes mmcv.utils importable
