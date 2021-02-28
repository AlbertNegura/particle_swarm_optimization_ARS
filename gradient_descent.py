"""Particle Swarm Optimization algorithm

Authors:
Julien Havel
Albert Negura
Sergi Nogues Farres (primary)
"""
import numpy as np
from utils import evaluate

def gradient_descent(function = "rosenbrock", iterations = 50, population=50):
    """

    :param function: (str, default="rosenbrock", options="rosenbrock" "rastrigin") the cost function to optimize
    :param iterations: (int, default=50) number of iterations or time steps the algorithm should simulate
    :param population: (int, default=50) number of particles in the gradient descent swarm
    :return: data, cost_function - a tuple of all results
             data: list of the form [x,y,[z1,z2]], where x is the time step, y is the Particle object id, [z1,z2] represent the [x,y] position tuple of the particle at the respective time step.
             cost_function: 1d list containing the minimum cost function value found by the swarm at each time step
    """
    
    # rastrigin does not converge with LR bigger than 0.005, yet it cannot find the global minimum but a local minimum
    learning_rate = 0.1 if function == "rosenbrock" else 0.005
    # same initialization as particle_swarm_optimization.py
    bounds = [-2.4, 2.4] if function == "rosenbrock" else [-5.12, 5.12]

    data = [[[0, 0] for _ in range(population)] for _ in range(iterations)]
    cost_function = np.zeros(iterations)
    cost_function_temp = np.zeros((population,iterations))
    x0 = np.min(bounds) + np.random.rand(population, 2) * (np.max(bounds) - np.min(bounds))
    for j in range(population):
        position = x0[j, :]
        for i in range(iterations):
            # main gradient descent
            cost = evaluate(function,position)
            position = step(function,position,learning_rate)

            for k in range(2):
                if position[k] > np.max(bounds):
                    position[k] = np.max(bounds)
                if position[k] < np.min(bounds):
                    position[k] = np.min(bounds)

            data[i][j][0] = position[0]
            data[i][j][1] = position[1]
            cost_function_temp[j][i] = cost
    # calculate the minimum cost among all gradient descent "particles"
    for i in range(iterations):
        cost_function[i] = np.min(cost_function_temp[:,i])
            #print("x {}, y {}, cost {}, iteration {}".format(position[0],position[1],cost,i))


    return data,cost_function


def step(function, position,learning_rate):
        """
        Single step in gradient descent.
        :param function: (str) Cost functions used (options: rastrigin rosenbrock)
        :param position: (list of the form [x,y]) x-y coordinates for individual particle to evaluate
        :param learning_rate: (float) gradient descent learning rate
        :return: list of the form [x,y] containing the next position of the particle
        """
        x = position[0]
        y = position[1]

        if function == "rastrigin":
            dx = 2*(x + 10*np.pi*np.sin(2*np.pi*x))
            dy = 2*(y + 10*np.pi*np.sin(2*np.pi*y))
            
        if function == "rosenbrock":
            dx = 2*(2*(x**3) - 2*x*y + x)
            dy = 2*(y - x**2)

        x_next = x - learning_rate*dx
        y_next = y - learning_rate*dy 
        return [x_next,y_next]


if __name__ == "__main__":
    gradient_descent()
