"""General reused snippets

Authors:
Julien Havel
Kamil Inglot
Albert Negura
Sergi Nogues Farres
"""
import numpy as np

def evaluate(function, position):
    """
    :param function: (str) Cost functions used (options: rastrigin rosenbrock)
    :param position: (list of the form [x,y]) x-y coordinates for individual particle to evaluate
    :return: cost of the particle based on the position
    """
    b = 1
    a = 0
    A = 10
    dimensions = 2

    x = position[0]
    y = position[1]

    if function == "rastrigin":  # rastrigin
        return A * dimensions + (x ** 2 - A * np.cos(np.pi * 2 * x)) + (y ** 2 - A * np.cos(np.pi * 2 * y))
    elif function == "rosenbrock":  # rosenbrock a=0,b=1
        return b* (y - x ** 2) ** 2 + (a - x) ** 2