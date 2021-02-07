import numpy as np
from particle import Particle


def pso(population=20, iterations=1000, a=0.5, b=0.5, c=0.5, optimize_a=True, function="rosenbrock", bounds=[-5.12, 5.12],
        neighbourhood="global"):
    if optimize_a:
        a_range = np.linspace(a, 0.4, iterations)
    else:
        a_range = [a]
    best_cost = np.inf
    pos_best_cost = []
    cost_function = np.zeros(iterations)

    if function == "rastrigin":  # rastrigin
        bounds = [-5.12, 5.12]
    elif function == "rosenbrock":  # rosenbrock a=0,b=1
        bounds = [-2.4, 2.4]

    x0 = np.min(bounds) + np.random.rand(population, 2) * (np.max(bounds) - np.min(bounds))
    swarm = [Particle(i, x0[i, :], population=population, neighbourhood=neighbourhood, bounds=bounds) for i in
             range(population)]
    position_matrix = [[[x0[i][0], x0[i][1]] for i in range(population)] for _ in range(iterations)]



    i = 0
    while i < iterations:
        for j in range(population):
            swarm[j].evaluate(function=function)
            if swarm[j].best_minimum_cost < best_cost:
                pos_best_cost = swarm[j].best_minimum_position
                best_cost = swarm[j].best_minimum_cost

        for j in range(population):
            if optimize_a:
                swarm[j].update_velocity(a_range[j], b, c, pos_best_cost, swarm)
            else:
                swarm[j].update_velocity(a_range[0], b, c, pos_best_cost, swarm)
            swarm[j].update_position()
            position_matrix[i][j][0] = swarm[j].position[0]
            position_matrix[i][j][1] = swarm[j].position[1]
        cost_function[i] = best_cost
        i += 1

    return position_matrix, cost_function


if __name__ == "__main__":
    pso()
