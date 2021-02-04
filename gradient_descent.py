import numpy as np
import random
import math

function = "rastrigin" 
bounds = [-2.4, 2.4] if function == "rosenbrock" else [-5.12, 5.12]
position = [random.uniform(bounds[0],bounds[1]),random.uniform(bounds[0],bounds[1])]
learning_rate = 0.01
iterations = 50

def evaluate(function, position):
        b = 1
        a = 0
        A = 10
        dimensions = 2

        x = position[0]
        y = position[1]

        if function == "rastrigin":  # rastrigin
            return A * dimensions + (x ** 2 - A * np.cos(math.pi * 2 * x)) + (y ** 2 - A * np.cos(math.pi * 2 * y))
        elif function == "rosenbrock":  # rosenbrock a=0,b=1
            return b*(y - x ** 2) ** 2 + (a-x) ** 2

def step(function, position):
        #b = 1
        #a = 0
        #A = 10
        #dimensions = 2

        x = position[0]
        y = position[1]

        if function == "rastrigin":
            dx = 2*(x + 10*math.pi*np.sin(2*math.pi*x))
            dy = 2*(y + 10*math.pi*np.sin(2*math.pi*y))
            
        if function == "rosenbrock":
            dx = 2*(2*(x**3) - 2*x*y + x)
            dy = 2*(y - x**2)

        x_next = x - learning_rate*dx
        y_next = y - learning_rate*dy 
        return [x_next,y_next]

data = [0 for x in range(iterations)]
cost = evaluate(function,position)
for i in range(iterations):
    data[i] = [position,cost]
    cost = evaluate(function,position)
    print("x {}, y {}, cost {}, iteration {}".format(position[0],position[1],cost,i))

    position = step(function,position)
        
