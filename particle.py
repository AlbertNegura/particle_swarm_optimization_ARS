import numpy as np
import random
import math

class Particle:
    position = []
    velocity = []
    best_minimum_position = []
    neighbourhood = []
    best_minimum_cost = None
    cost = None
    bounds = None

    def __init__(self, id, position, bounds=[-1,1], neighbourhood = "global", population = 0):
        self.position = position.copy()
        self.velocity = [random.uniform(bounds[0],bounds[1]),random.uniform(bounds[0],bounds[1])]
        self.best_minimum_position = position.copy()
        self.bounds = bounds
        if neighbourhood == "global":
            self.neighbourhood = [id]
        if neighbourhood == "geographical":
            self.neighbourhood = [id,id]
        if neighbourhood == "social-two": # 2 direct neighbours
            self.neighbourhood = [id-1,id,id+1]
            if id == 0:
                self.neighbourhood[0] = population-1
            if id == population-1:
                self.neighbourhood[2] = 0
        if neighbourhood == "social-four": # 4 direct neighbours
            self.neighbourhood = [id-2,id-1,id,id+1,id+2]
            if id == 0:
                self.neighbourhood[0] = population-2
                self.neighbourhood[1] = population-1
            if id == 1:
                self.neighbourhood[0] = population-1
            if id == population-1:
                self.neighbourhood[3] = 0
                self.neighbourhood[4] = 1
            if id == population-2:
                self.neighbourhood[4] = 0

    def evaluate(self, function, b=1, a=0, A=10, dimensions=2):
        x = self.position[0]
        y = self.position[1]

        if function == "rastrigin":  # rastrigin
            self.cost = A * dimensions + (x ** 2 - A * np.cos(math.pi * 2 * x)) + (y ** 2 - A * np.cos(math.pi * 2 * y))
            self.bounds = [-5.12, 5.12]
        elif function == "rosenbrock":  # rosenbrock a=0,b=1
            self.cost = b*(y - x ** 2) ** 2 + (a-x) ** 2
            self.bounds = [-2.4, 2.4]

    def update_velocity(self, a, b, c, pos_best_cost, swarm):
        r1 = random.random()
        r2 = random.random()
        if len(self.neighbourhood) == 1: # global
            for i in range(2):
                self.velocity[i] = a*self.velocity[i] + b*r1*(self.best_minimum_position[i]-self.position[i]) + c*r2*(pos_best_cost[i]-self.position[i])
                # if v(t+1) is larger, clip it to vmax
                if self.velocity[i] > 1:
                    self.velocity[i] = 1
                elif self.velocity[i] < -1:
                    self.velocity[i] = -1

        elif len(self.neighbourhood) == 2: # geographical
            swarm = np.asarray(swarm)
            best_neighbours = np.asarray([swarm[i].position-self.position for i in range(swarm.size)])
            best_neighbours = [best_neighbours[i].dot(best_neighbours[i]) for i in range(swarm.size)]
            best_neighbour = np.argsort(best_neighbours)
            best_neighbour = [swarm[best_neighbour[1]],swarm[best_neighbour[2]]]
            pos_best_cost = swarm[np.argmin([best_neighbour[0].cost,best_neighbour[1].cost])].position
            for i in range(2):
                self.velocity[i] = a*self.velocity[i] + b*r1*(self.best_minimum_position[i]-self.position[i]) + c*r2*(pos_best_cost[i]-self.position[i])
                # if v(t+1) is larger, clip it to vmax
                if self.velocity[i] > 1:
                    self.velocity[i] = 1
                elif self.velocity[i] < -1:
                    self.velocity[i] = -1

        elif len(self.neighbourhood) == 3: # social-two
            swarm = np.asarray(swarm)
            best_neighbours = [swarm[self.neighbourhood[0]],swarm[self.neighbourhood[2]]]
            pos_best_cost = swarm[np.argmin([best_neighbours[0].cost,best_neighbours[1].cost])].position
            for i in range(2):
                self.velocity[i] = a*self.velocity[i] + b*r1*(self.best_minimum_position[i]-self.position[i]) + c*r2*(pos_best_cost[i]-self.position[i])
                # if v(t+1) is larger, clip it to vmax
                if self.velocity[i] > 1:
                    self.velocity[i] = 1
                elif self.velocity[i] < -1:
                    self.velocity[i] = -1

        elif len(self.neighbourhood) == 5: # social-four
            swarm = np.asarray(swarm)
            best_neighbours = [swarm[self.neighbourhood[0]],swarm[self.neighbourhood[1]],swarm[self.neighbourhood[3]],swarm[self.neighbourhood[4]]]
            pos_best_cost = swarm[np.argmin([best_neighbours[0].cost,best_neighbours[1].cost,best_neighbours[2].cost,best_neighbours[3].cost])].position
            for i in range(2):
                self.velocity[i] = a*self.velocity[i] + b*r1*(self.best_minimum_position[i]-self.position[i]) + c*r2*(pos_best_cost[i]-self.position[i])
                # if v(t+1) is larger, clip it to vmax
                if self.velocity[i] > 1:
                    self.velocity[i] = 1
                elif self.velocity[i] < -1:
                    self.velocity[i] = -1


    def update_position(self):
        for i in range(2):
            self.position[i] = self.position[i] + self.velocity[i]
            if abs(self.position[i]) > abs(self.bounds[i]):
                self.position[i] = self.bounds[i]




if __name__ == "__main__":
    p = Particle()
