import numpy as np

def gradient_descent(function = "rosenbrock", iterations = 50, population=50):
    
    # rastrigin does not converge with LR bigger than 0.005, yet it cannot find the global minimum but a local minimum
    learning_rate = 0.1 if function == "rosenbrock" else 0.005
    bounds = [-2.4, 2.4] if function == "rosenbrock" else [-5.12, 5.12]

    data = [[[0, 0] for _ in range(population)] for _ in range(iterations)]
    cost_function = np.zeros(iterations)
    cost_function_temp = np.zeros((population,iterations))
    x0 = np.min(bounds) + np.random.rand(population, 2) * (np.max(bounds) - np.min(bounds))
    for j in range(population):
        position = x0[j, :]
        for i in range(iterations):
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
    for i in range(iterations):
        cost_function[i] = np.min(cost_function_temp[:,i])
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
            return A * dimensions + (x ** 2 - A * np.cos(np.pi * 2 * x)) + (y ** 2 - A * np.cos(np.pi * 2 * y))
        elif function == "rosenbrock":  # rosenbrock a=0,b=1
            return b*(y - x ** 2) ** 2 + (a-x) ** 2

def step(function, position,learning_rate):
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


gradient_descent()        
