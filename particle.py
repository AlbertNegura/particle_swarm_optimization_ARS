import numpy as np
import random
import math

class Particle:
    position = []
    velocity = []
    best_minimum_position = []
    best_minimum_cost = None
    cost = None

    def __init__(self, position, bounds=[-1,1]):
        self.position = position.copy()
        self.velocity = [random.uniform(bounds[0],bounds[1])]

    def evaluate(self, function):
        b = 1
        a = 0

        x = self.position[0]
        y = self.position[1]

        if function == "rastrigin":  # rastrigin
            self.cost = (x ** 2 - 10 * np.cos(math.pi * 2 * x ** 2)) + (y ** 2 - 10 * np.cos(math.pi * 2 * y ** 2))
        elif function == "rosenbrock":  # rosenbrock a=0,b=1
            self.cost = b*(y - x ** 2) ** 2 + (a-x) ** 2

    def update_velocity(self, a, b, c):
        r1 = random()
        r2 = random()

        # add velocity update

    def update_position(self):
        pass


if __name__ == "__main__":
    print("AAAAAAAA")