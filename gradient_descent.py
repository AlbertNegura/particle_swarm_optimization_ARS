import numpy as np
import math

def gradient_descent(function = "rosenbrock", iterations = 50):
    
    # rastrigin does not converge with LR bigger than 0.005, yet it cannot find the global minimum but a local minimum
    learning_rate = 0.1 if function == "rosenbrock" else 0.005
    bounds = [-2.4, 2.4] if function == "rosenbrock" else [-5.12, 5.12]

    x0 = np.min(bounds) + np.random.rand(1, 2) * (np.max(bounds) - np.min(bounds))

    position = x0[0, :]
    data = [[[0, 0, np.inf] for _ in range(1)] for _ in range(iterations)]
    cost_function = np.zeros(iterations)
    for i in range(iterations):
        cost = evaluate(function,position)
        position = step(function,position,learning_rate)

        for j in range(2):
            if position[j] > np.max(bounds):
                position[j] = np.max(bounds)
            if position[j] < np.min(bounds):
                position[j] = np.min(bounds)

        data[i][0][0] = position[0]
        data[i][0][1] = position[1]
        cost_function[i] = cost
        #print("x {}, y {}, cost {}, iteration {}".format(position[0],position[1],cost,i))


    return data,cost_function

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

def step(function, position,learning_rate):
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


gradient_descent()        
