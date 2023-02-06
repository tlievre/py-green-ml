"""
script testing
codecarbon_measurement script
"""
import pytest
# ------------REMOVE AS SOON AS POSSIBLE !!!!!!!!!!--------
import sys
sys.path.insert(0, "../")
# -----------REMOVE AS SOON AS POSSIBLE !!!!!!!!!!!--------
import codecarbon_measurement


@pytest.fixture
def fibonacci():
    n = 1000000
    a = 0
    b = 1
    # Check is n is less
    # than 0
    if n < 0:
        print("Incorrect input")
    # Check is n is equal
    # to 0
    elif n == 0:
        return 0
    # Check if n is equal to 1
    elif n == 1:
        return b
    else:
        for i in range(1, n):
            c = a + b
            a = b
            b = c
        return b


def test_codecarbon_measurement():
    codecarbon_obj = codecarbon_measurement.CodeCarbonMeasurement()
    codecarbon_obj.begin()
    _ = fibonacci
    codecarbon_obj.end()
    assert isinstance(codecarbon_obj.convert(), float) is True
