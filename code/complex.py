# Module computing complex numbers
# disclaimer : this class is not made to deal with less than 1e-10 values


from math import isclose
from typing import List, Union

from numpy import arctan2, cos, pi, sin, sqrt


class Complex:
    """Computing complex numbers"""

    def __init__(self, real=0.0, imaginary=0.0):
        self.re = real  # round(real, 15)
        self.im = imaginary  # round(imaginary,15)

    def __str__(self) -> str:
        if self.im == 0.0:
            string = f"{self.re}"
        elif self.re == 0:
            string = f"i({self.im})"
        else:
            string = f"{self.re} + i({self.im})"
        return string

    __repr__ = __str__

    def __eq__(self, other) -> bool:
        return bool(isclose(self.re, other.re) and isclose(self.im, other.im))

    def is_null(self):
        return isclose(self.re, 0) and isclose(self.im, 0)

    def is_real(self):
        return isclose(self.im, 0)

    def is_imaginary(self):
        return isclose(self.re, 0)

    def arg(self):
        """return the argument of the complex number
        return None if 0"""
        if self.is_null():
            arg = None
        elif isclose(self.re, 0) and self.im > 0:
            arg = pi / 2
        elif isclose(self.re, 0) and self.im < 0:
            arg = -pi / 2
        else:
            arg = round(arctan2(self.im, self.re), 15)
        return arg

    def module(self):
        """return the module of the complex number"""
        return round(sqrt(self.re**2 + self.im**2), 15)

    def conjuagate(self):
        return Complex(self.re, -self.im)

    # arithmetic
    def __add__(self, other):
        return Complex(self.re + other.re, self.im + other.im)

    def __sub__(self, other):
        return Complex(self.re - other.re, self.im - other.im)

    def __mul__(self, other):
        real = (self.re * other.re) - (self.im * other.im)
        imaginary = (self.re * other.im) + (self.im * other.re)
        return Complex(real, imaginary)

    def __truediv__(self, other):
        if other.is_null():
            raise ValueError("Error : dividing by 0")
        elif other.is_real():
            return Complex(self.re / other.re, self.im / other.re)
        else:
            denominator = (other.re**2) + (other.im**2)
            real = ((self.re * other.re) + (self.im * other.im)) / denominator
            imaginary = ((self.im * other.re) - (self.re * other.im)) / denominator
            return Complex(real, imaginary)


Num = Union[int, float]


def addition(
    *complexes: Complex,
) -> (
    Complex
):  # partially depreciated (can still be usefull for more iterable arguments)
    """calculate the sum of complex numbers

    parameters
    ----------
        - *complexes : iterable type of Complex

    return
    ------
        - sum of the complex numbers"""

    res = Complex(0)
    for number in complexes:
        res.re += number.re
        res.im += number.im
    return res


def difference(
    cpx1: Complex, cpx2: Complex = Complex(0)
):  # fully depreciated (replaced by __sub__ Complex methods)
    """calculate the difference of two complex numbers

    parameters
    ----------
        - cpx1 : Complex number
        - cpx2 : Complex number to subtract to cpx1 (=Complex(0) by default)

    return
    ------
        -   difference of the two complex numbers"""
    res = Complex()
    res.re = cpx1.re - cpx2.re
    res.im = cpx1.im - cpx2.im
    return res


def product(
    *complexes: Complex,
) -> (
    Complex
):  # partially depreciated (can still be usefull for more iterable arguments)
    """calculate the product of complex numbers

    parameters
    ----------
        - *complexes : iterable type of Complex

    return
    ------
        - product of the complex numbers"""
    res = Complex(1)
    for number in complexes:
        re = res.re * number.re - res.im * number.im
        im = res.re * number.im + res.im * number.re
        res.re = re
        res.im = im
    return res


def exp_to_literal(arg: float, module: float = 1.0) -> Complex:
    """return the literal expression of a complex number defined by its argument and module

    parameters
    ----------
        - arg : type(float) (should be between 0 and 2pi)
        - module : type(float) (must have a positive value)(=1 by default)

    return
    ------
        - Complex number associated"""
    assert module >= 0, "second-argument(module) must have a positive value"
    return Complex(module * cos(arg), module * sin(arg))


def nth_root(n: int, cpx: Complex = Complex(1)) -> Complex:
    """calculate the nth root of a complex number

    parameters
    ----------
        - n : type(int)
        - complex : type(Complex) (=Complex(1) by default) (must not be Complex(0))

    return
    ------
        - list of the nth roots"""
    assert (
        cpx.re != 0 or cpx.im != 0
    ), "second argument must be a non-zero complex number"
    module = cpx.module()
    arg = cpx.arg()
    if arg is not None:
        return exp_to_literal((arg / n), module ** (1 / n))
    else:
        return Complex(
            1
        )  # Not used case but just here to ensure nth_root cannot return None


def nth_roots_unity(n: int) -> list:
    """calculate the n roots of unity

    parameter
    ---------
        - n : type(int) : must be a positive integer

    return
    ------
        - a list of Complex containing the n roots of unity"""
    roots = [Complex(1) for i in range(n)]
    for k in range(0, n):
        roots[k] = exp_to_literal((2 * k * pi / n), 1.0)
    return roots


def inverse_nth_roots_unity(n: int) -> list:
    """calculate the inversed n roots of unity

    parameter
    ---------
    - n : type(int) : must be a positive integer

    return
    ------
    - a list of Complex containing the inversed n roots of unity"""
    roots = [Complex(1) for i in range(n)]
    for k in range(0, n):
        roots[k] = exp_to_literal((-2 * k * pi / n), 1.0)
    return roots


def make_complex(values: List[Num]) -> List[Complex]:
    res = []
    for value in values:
        res.extend([Complex(value)])
    return res


if __name__ == "__main__":
    pass
