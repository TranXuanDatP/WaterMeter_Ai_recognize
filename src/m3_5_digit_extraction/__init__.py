"""
M3.5: Black Digit Extraction Module

Extract 4 black digits (integer part) from M3 ROI images,
removing red digit (decimal part).
"""

from .extractor import M3_5_DigitExtractor

__all__ = ['M3_5_DigitExtractor']
