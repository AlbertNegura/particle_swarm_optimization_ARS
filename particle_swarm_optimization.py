"""Particle Swarm Optimization algorithm

Authors:
Julien Havel
Albert Negura
Sergi Nogues Farres
"""

import numpy as np
from particle import Particle


def pso(population=20, iterations=1000, a=0.5, b=0.5, c=0.5, optimize_a=True, function="rosenbrock", bounds=[-5.12, 5.12],
        neighbourhood="global"):
    """
    :param population: (int, default=20) number of particles in the swarm
    :param iterations: (int, default=1000) number of iterations or time steps the algorithm should simulate
    :param a: (float, default=0.5) the inertia constant
    :param b: (float, default=0.5) the social constant
    :param c: (float, default=0.5) the cognitive constant
    :param optimize_a: (boolean, default=True) whether to optimize the inertia constant
    :param function: (str, default="rosenbrock", options="rosenbrock" "rastrigin") the cost function to optimize
    :param bounds: (list of the form [x,y], optional) the bounds of the function
    :param neighbourhood: (str, default = "global", options = "global" "social-two" "social-four" "geographical") the neighbourhood type
    :return: position_matrix, cost_function, av_cost_function - a tuple of all results
             position_matrix: list of the form [x,y,[z1,z2]], where x is the time step, y is the Particle object id, [z1,z2] represent the [x,y] position tuple of the particle at the respective time step.
             cost_function: 1d list containing the minimum cost function value found by the swarm at each time step
             av_cost_function: 1d list containing the average cost function among all particles in the swarm at each time step
    """

    # whether to dynamically adjust a from the given value to 0.4
    if optimize_a:
        a_range = np.linspace(a, 0.4, iterations)
    else:
        a_range = [a]
    # initialize some variables / lists
    best_cost = np.inf
    pos_best_cost = []
    cost_function = np.zeros(iterations)
    av_cost_function = np.zeros(iterations)

    # set bounds depending on the function
    if function == "rastrigin":  # rastrigin
        bounds = [-5.12, 5.12]
    elif function == "rosenbrock":  # rosenbrock a=0,b=1
        bounds = [-2.4, 2.4]

    # initialize swarm
    x0 = np.min(bounds) + np.random.rand(population, 2) * (np.max(bounds) - np.min(bounds))
    swarm = [Particle(i, x0[i, :], population=population, neighbourhood=neighbourhood, bounds=bounds) for i in
             range(population)]
    position_matrix = [[[x0[i][0], x0[i][1]] for i in range(population)] for _ in range(iterations)]


    # iterate and update swarm at each step
    i = 0
    while i < iterations:
        cost_sum=0 # used for average cost plot
        # evaluate each member of the swarm separately, tracking if they found a better minimum
        for j in range(population):
            swarm[j].evaluate(function=function)
            if swarm[j].best_minimum_cost < best_cost:
                pos_best_cost = swarm[j].best_minimum_position
                best_cost = swarm[j].best_minimum_cost
            cost_sum += swarm[j].cost
        # update each member of the swarm separately for the next time step
        for j in range(population):
            if optimize_a:
                swarm[j].update_velocity(a_range[j], b, c, pos_best_cost, swarm)
            else:
                swarm[j].update_velocity(a_range[0], b, c, pos_best_cost, swarm)
            swarm[j].update_position()
            # store values for plotting
            position_matrix[i][j][0] = swarm[j].position[0]
            position_matrix[i][j][1] = swarm[j].position[1]
        cost_function[i] = best_cost
        av_cost_function[i] = cost_sum/population
        i += 1

    return position_matrix, cost_function, av_cost_function


if __name__ == "__main__":
    pso()
