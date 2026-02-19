import pytest
from main import add, sub, multiply, divide


def test_add():
    assert add(1, 2) == 3
    assert add(-1, 2) == 1


def test_subtract():
    assert sub(10, 5) == 5
    assert sub(-2, -4) == -2


@pytest.mark.xfail(raises=ZeroDivisionError)
def test_divide_by_zero():
    divide(10, 0)


def test_multiply():
    assert multiply(10, 5) == 50
    assert multiply(-2, -4) == 8


@pytest.mark.parametrize(
    "input1, input2, expected", [(3, 7, 10), (-6, 4, -2), (9, 0, 9)]
)
def test_add_nominal(input1, input2, expected):
    assert add(input1, input2) == expected


@pytest.mark.parametrize("input1, input2", [("3", 7), (9, "0"), ([], 8), (7, ["a"])])
def test_add_edge_cases(input1, input2):
    with pytest.raises((TypeError, ValueError)):
        add(input1, input2)


def add(x, y):
    return x + y


def subtract(x, y):
    if y > x:
        raise ValueError("y cannot be greater than x")
    return x - y


def test_subtract_nominal():
    assert subtract(5, 2) == 3


@pytest.mark.parametrize("x, y", [(0, 1), (1, -1)])
def test_subtract_edge(x, y):
    with pytest.raises(ValueError):
        subtract(x, y)
