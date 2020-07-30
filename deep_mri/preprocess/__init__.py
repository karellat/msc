"""
Utilities for preprocessing raw MRI (nifty images). It includes coregistration tools, bias filtering and brain extraction.

It uses mainly FLS and MINC toolkit.
"""

from .adni import get_adni_image_id
__all__ = ["get_adni_image_id"]
