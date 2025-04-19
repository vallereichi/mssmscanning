"""
Objective functions
"""

import numpy as np


type Vector = list[float]


def de_jong(point: Vector) -> float:
    """spheres summed"""
    return np.sum([x**2 for x in point])


def gaussian(point: Vector, mu: float = 50, sigma: float = 30) -> float:
    """
    negative log likelihood in the form of a gaussian shell
    """

    return np.sum([-0.5 * np.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2)) * (x - mu) ** 2 for x in point])


def parabolic(point: Vector) -> float:
    """
    test objective function
    """

    return point[0] ** 2 + point[1] ** 2


def two_valleys(point: Vector) -> float:
    """
    test objective function
    """
    x = point[0]
    y = point[1]
    valley1 = 5 * np.exp(-((x - 20) ** 2 + (y - 2) ** 2) / 100)
    valley2 = 10 * np.exp(-((x + 20) ** 2 + (y + 2) ** 2) / 300)

    return -valley1 - valley2 + 100


def four_valleys(point: Vector) -> float:
    """
    test objective function
    """
    x = point[0]
    y = point[1]
    valley1 = np.exp(-((x - 20) ** 2 + (y - 20) ** 2) / 200)
    valley2 = np.exp(-((x + 20) ** 2 + (y - 20) ** 2) / 200)
    valley3 = np.exp(-((x - 20) ** 2 + (y + 20) ** 2) / 200)
    valley4 = np.exp(-((x + 20) ** 2 + (y + 20) ** 2) / 200)
    return -valley1 - valley2 - valley3 - valley4 + 1000
