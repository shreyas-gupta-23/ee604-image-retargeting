"""Spatial-domain image retargeting. We stay in the pixel grid and do not train a retargeting network."""

from .retarget import retarget

__all__ = ["retarget"]
