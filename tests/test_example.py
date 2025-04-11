"""This is a sample test module.
Replace its content with the tests for your plugin.

Run the tests from the root of the repository with:
>>> pytest -v tests/
"""

import pytest


# Basic test cases
def test_foo():
    """Test if 'foo' is a string."""
    assert isinstance("foo", str)


def test_bar():
    """Test if the length of 'bar' is 3."""
    assert len("bar") == 3


# Testing numbers
def test_sum():
    """Test simple sum operation."""
    assert 2 + 3 == 5


def test_division():
    """Test division operation."""
    assert 10 / 2 == 5


# Testing exceptions
def test_divide_by_zero():
    """Test that division by zero raises ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        1 / 0


# Using pytest.mark.parametrize for multiple test cases
@pytest.mark.parametrize(
    "input_string,expected_length",
    [
        ("hello", 5),
        ("pytest", 6),
        ("", 0),
    ],
)
def test_string_length(input_string, expected_length):
    """Test multiple string lengths."""
    assert len(input_string) == expected_length
