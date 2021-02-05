import numpy as np
from particle import Particle

def pso(population=20, iterations=50, a=0.9, b=2.0, c=2.0, optimize_a=True,function="rosenbrock", bounds=[-5.12,5.12], neighbourhood="global"):
    if optimize_a:
        a_range = np.linspace(a, 0.4, iterations)
    else:
        a_range = [0]
    best_cost = np.inf
    pos_best_cost = []

    if function == "rastrigin":  # rastrigin
        bounds = [-5.12, 5.12]
    elif function == "rosenbrock":  # rosenbrock a=0,b=1
        bounds = [-2.4, 2.4]

    x0 = np.random.uniform(low=bounds[0], high=bounds[1], size=(population,2))

    swarm = [Particle(i,x0[i,:],population=population,neighbourhood=neighbourhood) for i in range(population)]
    position_matrix = [[[0,0, np.inf] for _ in range(population)] for _ in range(iterations)]

    i = 0
    while i < iterations:
        for j in range(population):
            swarm[j].evaluate(function=function)

            if swarm[j].cost<best_cost:
                pos_best_cost = swarm[j].position
                best_cost = swarm[j].cost

        for j in range(population):
            if optimize_a:
                swarm[j].update_velocity(a_range[j], b, c, pos_best_cost, swarm)
            else:
                swarm[j].update_velocity(a, b, c, pos_best_cost, swarm)
            swarm[j].update_position()
            position_matrix[i][j][0] = swarm[j].position[0]
            position_matrix[i][j][1] = swarm[j].position[1]
        i+=1

    return position_matrix


if __name__ == "__main__":
    pso()