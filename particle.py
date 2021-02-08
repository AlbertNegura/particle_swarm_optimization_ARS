"""Particle Class corresponding to Particle Swarm Optimization

Authors:
Julien Havel
Albert Negura
Sergi Nogues Farres
"""
import numpy as np

class Particle:
    position = []
    velocity = []
    best_minimum_position = []
    neighbourhood = []
    best_minimum_cost = np.inf
    cost = np.inf
    bounds = None
    id = None

    def __init__(self, id, position, bounds=None, neighbourhood ="global", population = 0):
        """
        Args:
            id (int): A global id for the particle
            position (list): A 2d list corresponding to the position of the particle of the form [x,y]
            bounds (list): A 2d list corresponding to the (square area) bounds of the cost function of the form [min,max]
            neighbourhood (str, optional): Type of neighbourhood used (options: global social-two social-four geographical)
            population (int, optional): Number of particles in the swarm
        """
        if bounds is None:
            bounds = [-1, 1]
        self.id = id
        self.bounds = bounds
        self.position = position.copy()
        # calculate velocity bounds based on spatial bounds
        self.upper_bound_vel = np.abs(np.max(self.bounds)-np.min(self.bounds))
        self.lower_bound_vel = -self.upper_bound_vel
        # overwrite velocity bounds for better visualizations
        self.upper_bound_vel = 0.1
        self.lower_bound_vel = -0.1
        # initialize velocity with random components within velocity bounds
        self.velocity = self.lower_bound_vel + np.random.rand(2) * (self.upper_bound_vel-self.lower_bound_vel)
        # set neighbourhood
        if neighbourhood == "global": # just ignore
            self.neighbourhood = [id]
        elif neighbourhood == "geographical": # 2 nearest neighbours
            self.neighbourhood = [id,id]
        elif neighbourhood == "social-two": # 2 direct social neighbours
            self.neighbourhood = [id-1,id,id+1]
            if id == 0:
                self.neighbourhood[0] = population-1
            if id == population-1:
                self.neighbourhood[2] = 0
        elif neighbourhood == "social-four": # 4 direct social neighbours
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
        """
        Sets the cost of the particle based on its current position.
        Args:
            function (str): Cost functions used (options: rastrigin rosenbrock)
            b (int, optional): Rosenbrock function x-y component
            a (int, optional): Rosenbrock function a component (defined minimum area)
            A (int, optional): Rastrigin function constant
            dimensions (int, optional): Number of dimensions (options: 2)
        """
        x = self.position[0]
        y = self.position[1]

        if function == "rastrigin":  # rastrigin
            self.cost = A * dimensions + (x ** 2 - A * np.cos(np.pi * 2 * x)) + (y ** 2 - A * np.cos(np.pi * 2 * y))
        elif function == "rosenbrock":  # rosenbrock a=0,b=1
            self.cost = b*(y - x ** 2) ** 2 + (a-x) ** 2
        if self.cost < self.best_minimum_cost:
            self.best_minimum_cost = self.cost
            self.best_minimum_position = self.position

    def update_velocity(self, a, b, c, pos_best_cost, swarm):
        """
        Sets the velocity of the particle based on its current position as well as its best minimum position and the best swarm minimum position.
        Args:
            a (float): Omega / Inertia constant
            b (float): Social constant
            c (float): Cognitive constant
            pos_best_cost (float): Best minimum cost found by swarm
            swarm (list): A list with all members of the swarm
        """
        # generate some randomization factors for the acceleration vectors
        r1 = np.random.uniform(low=0., high=1.0, size=(2))
        r2 = np.random.uniform(low=0., high=1.0, size=(2))
        if len(self.neighbourhood) == 1: # global
            for i in range(2):
                self.velocity[i] = a*self.velocity[i] + c*r1[i]*(self.best_minimum_position[i]-self.position[i]) + b*r2[i]*(pos_best_cost[i]-self.position[i])
                # if v(t+1) is larger, clip it to vmax
                if self.velocity[i] > self.upper_bound_vel:
                    self.velocity[i] = self.upper_bound_vel
                elif self.velocity[i] < self.lower_bound_vel:
                    self.velocity[i] = self.lower_bound_vel

        elif len(self.neighbourhood) == 2: # geographical
            #determine closest two neighbours
            swarm = np.asarray(swarm)
            best_neighbours = np.asarray([swarm[i].position-self.position for i in range(swarm.size)])
            best_neighbours = [best_neighbours[i].dot(best_neighbours[i]) for i in range(swarm.size)]
            best_neighbour = np.argsort(best_neighbours)
            best_neighbour = [swarm[best_neighbour[1]],swarm[best_neighbour[2]]]
            pos_best_cost = swarm[np.argmin([best_neighbour[0].cost,best_neighbour[1].cost])].position
            # update position based on best between closest 2 neighbours
            for i in range(2):
                self.velocity[i] = a*self.velocity[i] + c*r1*(self.best_minimum_position[i]-self.position[i]) + b*r2*(pos_best_cost[i]-self.position[i])
                # if v(t+1) is larger, clip it to vmax
                if self.velocity[i] > self.upper_bound_vel:
                    self.velocity[i] = self.upper_bound_vel
                elif self.velocity[i] < self.lower_bound_vel:
                    self.velocity[i] = self.lower_bound_vel

        elif len(self.neighbourhood) == 3: # social-two
            swarm = np.asarray(swarm)
            best_neighbours = [swarm[self.neighbourhood[0]],swarm[self.neighbourhood[2]]]
            pos_best_cost = swarm[np.argmin([best_neighbours[0].cost,best_neighbours[1].cost])].position
            # update position based on best between 2 social neighbours
            for i in range(2):
                self.velocity[i] = a*self.velocity[i] + c*r1*(self.best_minimum_position[i]-self.position[i]) + b*r2*(pos_best_cost[i]-self.position[i])
                # if v(t+1) is larger, clip it to vmax
                if self.velocity[i] > self.upper_bound_vel:
                    self.velocity[i] = self.upper_bound_vel
                elif self.velocity[i] < self.lower_bound_vel:
                    self.velocity[i] = self.lower_bound_vel

        elif len(self.neighbourhood) == 5: # social-four
            swarm = np.asarray(swarm)
            best_neighbours = [swarm[self.neighbourhood[0]],swarm[self.neighbourhood[1]],swarm[self.neighbourhood[3]],swarm[self.neighbourhood[4]]]
            pos_best_cost = swarm[np.argmin([best_neighbours[0].cost,best_neighbours[1].cost,best_neighbours[2].cost,best_neighbours[3].cost])].position
            # update position based on best between 4 social neighbours
            for i in range(2):
                self.velocity[i] = a*self.velocity[i] + c*r1*(self.best_minimum_position[i]-self.position[i]) + b*r2*(pos_best_cost[i]-self.position[i])
                # if v(t+1) is larger, clip it to vmax
                if self.velocity[i] > self.upper_bound_vel:
                    self.velocity[i] = self.upper_bound_vel
                elif self.velocity[i] < self.lower_bound_vel:
                    self.velocity[i] = self.lower_bound_vel


    def update_position(self):
        """
        Updates the position of the particles based on its velocity, bounded by the bounds of the cost function.
        """
        for i in range(2):
            self.position[i] = self.position[i] + self.velocity[i]
            if self.position[i] > np.max(self.bounds):
                self.position[i] = np.max(self.bounds)
            if self.position[i] < np.min(self.bounds):
                self.position[i] = np.min(self.bounds)




if __name__ == "__main__":
    p = Particle()
