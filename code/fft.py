# fast-fourier transforms

from cmath import cos, exp, pi

import complex as cpx
from numpy import log2


def FFT(vector: list) -> list:
    """calculate the fast fourier tranform of a vector

    parameters
    ----------
        -vector : list of Complex object

    return
    ------
        - 1-D fast fourier transform of the vector"""
    n = len(vector)
    assert log2(
        n
    ).is_integer(), "make sure that the length of the arguement is a power of 2"
    if n == 1:
        return vector
    poly_even, poly_odd = vector[::2], vector[1::2]
    res_even, res_odd = FFT(poly_even), FFT(poly_odd)
    res = [cpx.Complex(0)] * n
    for j in range(n // 2):
        w_j = cpx.exp_to_literal(-2 * pi * j / n)
        product = w_j * res_odd[j]
        res[j] = res_even[j] + product
        res[j + n // 2] = res_even[j] - product
    return res


def IFFT_aux(vector: list) -> list:
    """auxiliary function that makes the recursive steps of the IFFT algorithm
    parameters
    ----------
        -vector : list of Complex object

    return
    ------
        - partial inverse of the 1-D fast fourier transform of the vector (lack the division by n)
    """
    n = len(vector)
    assert log2(
        n
    ).is_integer(), "make sure that the length of the arguement is a power of 2"
    if n == 1:
        return vector
    poly_even, poly_odd = vector[::2], vector[1::2]
    res_even, res_odd = IFFT_aux(poly_even), IFFT_aux(poly_odd)
    res = [cpx.Complex(0)] * n
    for j in range(n // 2):
        w_j = cpx.exp_to_literal((2 * pi * j) / n)
        product = w_j * res_odd[j]
        res[j] = res_even[j] + product
        res[j + n // 2] = res_even[j] - product
    return res


def IFFT(vector: list) -> list:
    """caclulate the inverse of the fast fourier tranform of a vector (in order to have ifft(fft(poly)) == poly)

    parameters
    ----------
        -vector : list of Complex object

    return
    ------
        - inverse of the 1-D fast fourier transform of the vector"""
    n = len(vector)
    res = IFFT_aux(vector)
    for i in range(n):
        res[i] = res[i] / cpx.Complex(n)
    return res
