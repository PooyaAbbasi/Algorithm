""" some utility tools for algorithms implementation
"""
import math


def is_power_of_2(n: int) -> bool:
    if n <= 0:
        return False
    else:
        return (n & (n - 1)) == 0


def get_next_power_of_two_greater_than(n: int) -> int:
    return 2 ** math.ceil(math.log2(n))

