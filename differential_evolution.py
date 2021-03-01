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

def cross_over(dad, mom):
    if np.random.rand() > .5:
        return [dad[0], mom[1]]
    else:
        return [mom[0], dad[1]]

def mutate(parent, lower_bound, upper_bound):
    if np.random.rand() > .5:
        return [np.random.uniform(lower_bound, upper_bound), parent[1]]
    else:
        return [parent[0], np.random.uniform(lower_bound, upper_bound)]



def genetic_algorithm(mutation = 0.1, population = 10, function="rosenbrock", max_iterations=5000, target_fitness=None, dimensions = 2, selection="elitism"):
    """
    :param crossover:
    :param mutation:
    :param selection:
    :param pop_size:
    :return:

    - init agents x at random positions in search space
    - calculate fitness value of each agent
    - while(not end) : (end can be number of iterations, adequate fitness reached, ...)
        - for each agent :
            - calculate whether selected (selection strategy)
        - for each agent :
            - if selected, use as parent
            - otherwise, calculate cross-over with parents based on selection and mutation
            - calculate fitness value in the new iteration

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
    fitness_history = np.zeros((max_iterations,len(agents)),dtype=float)
    fitness_history[0] = [evaluate(function, agent) for agent in agents]
    num_selected = int(population/5) if selection == "elitism" or selection == "roulette" else int(population/5*4) if selection == "steady" else int(population/2)
    while not end:
        iteration_history = []
        selected_agents = []
        # SELECTION
        if selection == "elitism" or selection == "steady":
            selected_agents = np.argpartition(fitness_history[iteration], num_selected)
        elif selection == "tournament":
            random_order = np.random.choice(range(population),population, replace=False)
            left_bracket = random_order[:num_selected]
            right_bracket = random_order[num_selected:]
            for i in range(num_selected):
                selected_agent = left_bracket[i] if fitness_history[iteration][left_bracket[i]] > fitness_history[iteration][right_bracket[i]]  else right_bracket[i] if fitness_history[iteration][left_bracket[i]] < fitness_history[iteration][right_bracket[i]] else np.random.choice([left_bracket[i],right_bracket[i]])
                selected_agents.append(selected_agent)
        elif selection == "roulette":
            total_fitness = np.sum(fitness_history[iteration])
            random_order = np.random.choice(range(population), population, replace=False)
            roulette_selection = {int(i) : fitness_history[iteration][int(i)] for i in random_order}
            chance = np.random.uniform(0, total_fitness)
            i = 0
            current = 0
            for key, value in roulette_selection.items():
                current += value
                if current > chance:
                    selected_agents.append(key)
                    i+=1
                    if i >= num_selected:
                        break
        else: # in case of error, default to elitism
            selected_agents = np.argpartition(fitness_history[iteration], num_selected)
        for x in range(len(agents)):
            if x in selected_agents[:num_selected]:
                new_agent = agents[x]
            else:
                # CROSSOVER
                if len(selected_agents) > 0:
                    dad = np.random.choice(selected_agents)
                    mom = np.random.choice(selected_agents)
                    while dad == mom:
                        mom = np.random.choice(selected_agents)
                else: # Roulette selection is special
                    parents = np.random.choice(range(population),2,replace=False)
                    dad = parents[0]
                    mom = parents[1]
                new_agent = cross_over(agents[dad],agents[mom])
                # MUTATION
                if np.random.rand() < mutation:
                    new_agent = mutate(new_agent, bounds[0], bounds[1])

            agents[x] = new_agent
            iteration_history.append(new_agent)

        agents_history[iteration] = np.clip(iteration_history,bounds[0],bounds[1])
        fitness_history[iteration] = [evaluate(function, agent) for agent in agents]
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