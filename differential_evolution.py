"""Particle Swarm Optimization algorithm

Authors:
Julien Havel
"""
import random
import numpy as np
from utils import evaluate



def differential_evolution(crossover = 0.9, differential_weight = 0.8, population = 10, function="rosenbrock", max_iterations=5000, target_fitness=None, dimensions = 2):
    # check crossover in [0,1]
    # check weight in [0,2]
    # check pop >= 4
    # check either max_iter or target_fitness != None (I get both can be.. Untested for now)
    # check dimensions > 0 and an Integer

    """
    :param crossover:
    :param differential_weight:
    :param pop_size:
    :return:

    - init agents x at random positions in search space
    - while(not end) : (end can be number of iterations, adequate fitness reached, ...)
        - for each agent :
            - pick 3 other agents at random
            - pick a random index (bounded by size of problem)

            - for each index :
                - pick a uniformly distributed random number ri ~ U(0,1)
                - if (ri < crossover) or (i = R ):
                    - yi = ai + F x (bi - ci)
                - else :
                    - yi = xi

            - if F(y) <= F(x) :
                - replace agent x with the improved (or equal) agent y

    - Pick agent with best fitness and return it as solution

    """
    bounds = [-2.4, 2.4] if function == "rosenbrock" else [-5.12, 5.12]

    agents = [[np.random.uniform(low=bounds[0], high=bounds[1]) for _ in range(dimensions)] for _ in range(population)]
    for agent in agents:
        np.clip(agent, bounds[0], bounds[1])
    end = False
    iteration = 0
    best_fitness = None
    agents_history = np.zeros((max_iterations,len(agents),2),dtype=float)
    while not end:
        iteration_history = []

        for x in range(len(agents)):
            new_agent = [0 for _ in range(dimensions)]
            temp = []
            while len(temp) < 3:
                rand_agent = random.randint(0, len(agents)-1)
                if (rand_agent not in temp) or rand_agent != x:
                    temp.append(rand_agent)
            a = temp[0]
            b = temp[1]
            c = temp[2]
            rand_index = 1 if dimensions < 2 else random.randint(1, dimensions)

            for index in range(dimensions):
                ri = np.random.uniform(0.0, 1.0)
                if (ri < crossover) or (index == rand_index):
                    new_agent[index] = agents[a][index] + differential_weight * (agents[b][index] - agents[c][index])
                else:
                    new_agent[index] = agents[x][index]
            if evaluate(function, new_agent) <= evaluate(function, agents[x]):
                agents[x] = new_agent
            iteration_history.append(new_agent)

        agents_history[iteration] = np.clip(iteration_history,bounds[0],bounds[1])
        iteration += 1
        if (max_iterations != None) and (max_iterations == iteration):
            end = True
        if (target_fitness != None) and (target_fitness == best_fitness):
            end = True

    # calculate total fitness for all agents
    fitnesses = np.array([[evaluate(function, agent) for agent in agents] for agents in agents_history])
    sum_fitness = [np.sum(fitnesses[:,i]) for i in range(population)]
    best_agent_id = np.argmin(sum_fitness)
    best_agent = agents_history[:,best_agent_id,:]
    best_fitness = fitnesses[:,best_agent_id]
    return best_agent, best_fitness, agents_history #best_fitness_history

def genetic_algorithm(crossover = 0.9, mutation = 0.1, population = 10, function="rosenbrock", max_iterations=5000, target_fitness=None, dimensions = 2):
    # check crossover in [0,1]
    # check weight in [0,2]
    # check pop >= 4
    # check either max_iter or target_fitness != None (I get both can be.. Untested for now)
    # check dimensions > 0 and an Integer

    """
    :param crossover:
    :param differential_weight:
    :param pop_size:
    :return:

    - init agents x at random positions in search space
    - while(not end) : (end can be number of iterations, adequate fitness reached, ...)
        - for each agent :
            - pick 3 other agents at random
            - pick a random index (bounded by size of problem)

            - for each index :
                - pick a uniformly distributed random number ri ~ U(0,1)
                - if (ri < crossover) or (i = R ):
                    - yi = ai + F x (bi - ci)
                - else :
                    - yi = xi

            - if F(y) <= F(x) :
                - replace agent x with the improved (or equal) agent y

    - Pick agent with best fitness and return it as solution

    """
    bounds = [-2.4, 2.4] if function == "rosenbrock" else [-5.12, 5.12]

    agents = [[np.random.uniform(low=bounds[0], high=bounds[1]) for _ in range(dimensions)] for _ in range(population)]
    for agent in agents:
        np.clip(agent, bounds[0], bounds[1])
    end = False
    iteration = 0
    best_fitness = None
    agents_history = np.zeros((max_iterations,len(agents),2),dtype=float)
    while not end:
        iteration_history = []

        for x in range(len(agents)):
            new_agent = [0 for _ in range(dimensions)]
            temp = []
            while len(temp) < 3:
                rand_agent = random.randint(0, len(agents)-1)
                if (rand_agent not in temp) or rand_agent != x:
                    temp.append(rand_agent)
            a = temp[0]
            b = temp[1]
            c = temp[2]
            rand_index = 1 if dimensions < 2 else random.randint(1, dimensions)

            for index in range(dimensions):
                ri = np.random.uniform(0.0, 1.0)
                if (ri < crossover) or (index == rand_index):
                    new_agent[index] = agents[a][index] + differential_weight * (agents[b][index] - agents[c][index])
                else:
                    new_agent[index] = agents[x][index]
            if evaluate(function, new_agent) <= evaluate(function, agents[x]):
                agents[x] = new_agent
            iteration_history.append(new_agent)

        agents_history[iteration] = np.clip(iteration_history,bounds[0],bounds[1])
        iteration += 1
        if (max_iterations != None) and (max_iterations == iteration):
            end = True
        if (target_fitness != None) and (target_fitness == best_fitness):
            end = True

    # calculate total fitness for all agents
    fitnesses = np.array([[evaluate(function, agent) for agent in agents] for agents in agents_history])
    sum_fitness = [np.sum(fitnesses[:,i]) for i in range(population)]
    best_agent_id = np.argmin(sum_fitness)
    best_agent = agents_history[:,best_agent_id,:]
    best_fitness = fitnesses[:,best_agent_id]
    return best_agent, best_fitness, agents_history #best_fitness_history



if __name__ == "__main__":
    best_agent, best_fitness, agents_history = differential_evolution()
    print(evaluate("rosenbrock", best_agent))