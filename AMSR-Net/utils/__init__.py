"""
utils/__init__.py
==================
Public API for the AMSR-Net utils package.
"""

from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    best_ckpt_path,
    latest_ckpt_path,
    epoch_ckpt_path,
)

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "best_ckpt_path",
    "latest_ckpt_path",
    "epoch_ckpt_path",
    "Trainer",
]


def __getattr__(name):
    if name == "Trainer":
        from utils.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
