"""
Math Functions:
A collection of basic mathematical operations including addition and multiplication.
"""

from openai_decorator import openaifunc

@openaifunc
def sum(a: int, b: int) -> int:
    """
    This function adds two numbers.
    :param a: The first number to add
    :param b: The second number to add
    """
    return a + b

@openaifunc
def multiply(a: int, b: int) -> int:
    """
    This function multiplies two numbers.
    :param a: The first number to multiply
    :param b: The second number to multiply
    """
    return a * b
