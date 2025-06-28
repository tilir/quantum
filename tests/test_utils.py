"""
Test suite for filename validation utilities.

This module contains tests for the filename validation and processing functions
from experiments.utils. The tests verify both normal operation and edge cases
to ensure robust handling of filenames in the quantum computing experiments pipeline.

Key Test Cases:
- Automatic extension addition for missing .png
- Proper handling of already valid filenames
- Empty filename error handling
"""

import pytest

from experiments.utils import validate_filename


def test_validate_filename():
    """
    Test the validate_filename function which ensures proper image filenames.

    The function should:
    1. Automatically append .png extension when missing
    2. Preserve correctly formatted filenames
    3. Reject empty filenames with system exit

    These tests verify the function meets quantum experiment requirements where:
    - Consistent image formats are crucial for data processing
    - Invalid filenames could disrupt experiment result collection
    """

    # Test default extension addition behavior
    assert (
        validate_filename("test") == "test.png"
    ), "Default case: Should append .png when extension is missing"

    # Test preservation of properly formatted filenames
    assert (
        validate_filename("image.png") == "image.png"
    ), "Already valid: Should preserve correct .png filenames unchanged"

    # Test empty filename rejection
    with pytest.raises(SystemExit):
        validate_filename(
            ""
        ), "Error case: Should system exit on empty filenames to prevent data loss"
