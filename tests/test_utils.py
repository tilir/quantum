import pytest
import sys
from pathlib import Path
from experiments.utils import validate_filename

def test_validate_filename():
    assert validate_filename("test") == "test.png"
    assert validate_filename("image.png") == "image.png"
    with pytest.raises(SystemExit):
        validate_filename("")
