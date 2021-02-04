import particle_swarm_optimization as pso
import particle
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib import cm


matplotlib.use("TkAgg")

cost = []
bounds = []

def plot_all(positions, cost, function):
    global bounds
    bounds = [-2.4, 2.4] if function == "rosenbrock" else [-5.12, 5.12]
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    contour_plot(positions, function, ax1)
    surface_plot(positions, function, ax2)
    cost_plot(cost, function, ax3)
    plt.show()

def function_of(x, y, function):
    b = 1
    a = 0
    A = 10
    dimensions = 2

    if function == "rastrigin":  # rastrigin
        return A * dimensions + (x ** 2 - A * np.cos(math.pi * 2 * x)) + (y ** 2 - A * np.cos(math.pi * 2 * y))
    elif function == "rosenbrock":  # rosenbrock a=0,b=1
        return b * (y - x ** 2) ** 2 + (a - x) ** 2


def contour_plot(data, function, ax):
    x = np.arange(np.min(bounds), np.max(bounds), 0.05)
    y = np.arange(np.min(bounds), np.max(bounds), 0.05)

    X, Y = np.meshgrid(x, y)
    zs = np.array(function_of(np.ravel(X), np.ravel(Y), function))
    Z = zs.reshape(X.shape)

    ax.contour(X, Y, Z, levels=50, cmap='viridis')
    ax.title.set_text("2D Contour Plot of Objective Function")

    xs = data[:,:,0]
    ys = data[:,:,1]

def surface_plot(data, function, ax):
    x = np.arange(np.min(bounds), np.max(bounds), 0.05)
    y = np.arange(np.min(bounds), np.max(bounds), 0.05)

    X, Y = np.meshgrid(x, y)
    zs = np.array(function_of(np.ravel(X), np.ravel(Y), function))
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap=cm.nipy_spectral, alpha=0.6)
    #ax.contour3D(X, Y, Z, 50, cmap='gray', linestyles="solid")
    ax.title.set_text("3D Plot of Objective Function")

def cost_plot(data, function, ax):
    ax.set(xlim=bounds, ylim=bounds)
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)
    #indices = np.linspace(0, data.len(), self.stop - 1)
    # ax.xlabel.set_text('Iterations')
    # ax.ylabel.set_text('Cost')
    ax.title.set_text('Min Cost Function')
    ax.plot(data, lw=3)

#ani3 = animation.FuncAnimation(fig, animate3, frames=iterations, fargs=[],interval=100, blit=False, repeat=True)

if __name__ == "__main__":
    #plot_all([],[],"rastrigin")
    data=pso.pso()
    plot_all(np.asarray(data), [], "rosenbrock")