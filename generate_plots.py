import particle_swarm_optimization as pso
import gradient_descent as gd
import numpy as np
import math
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib import style
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk

LARGE_FONT = ("Verdana", 12)
style.use("ggplot")

cost = []
bounds = []
fig = plt.figure()


def plot_all(positions, cost, function):
    global bounds, fig
    bounds = [-2.4, 2.4] if function == "rosenbrock" else [-5.12, 5.12]
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    contour_plot(positions, function, ax1)
    surface_plot(positions, function, ax2)
    cost_plot(cost, function, ax3)
    plt.show()


def stop_animations(ani1, ani2):
    ani1.event_source.stop()
    ani2.event_source.stop()


def start_animations(ani1, ani2):
    ani1.event_source.start()
    ani2.event_source.start()


def recalculate():
    # data,cost=pso.pso(function="rosenbrock", optimize_a=True, a=0.9, b=2.0, c=2.0)
    # gui(np.asarray(data), cost, "rosenbrock")
    pass


def gui(positions, cost, function):
    global bounds, f1, f2, f3
    bounds = [-2.4, 2.4] if function == "rosenbrock" else [-5.12, 5.12]
    root = tk.Tk()

    f1 = plt.Figure(dpi=100)
    ax1 = f1.add_subplot(111)
    l1 = FigureCanvasTkAgg(f1, root)
    l1.get_tk_widget().grid(row=0, column=0)
    cont_data, cont_scatters, cont_lines = contour_plot(positions, function, ax1)

    #

    f2 = plt.Figure(dpi=100)
    ax2 = f2.add_subplot(111, projection='3d')
    l2 = FigureCanvasTkAgg(f2, root)
    l2.get_tk_widget().grid(row=0, column=1)
    surf_data, surf_zs, surf_scatters, surf_lines = surface_plot(positions, function, ax2)

    ani1 = animation.FuncAnimation(f1, animate_contour, frames=positions.shape[0],
                                   fargs=[cont_data, cont_scatters, cont_lines], interval=10,
                                   blit=False, repeat=True)
    ani2 = animation.FuncAnimation(f2, animate_surface, frames=positions.shape[0],
                                   fargs=[surf_data, surf_zs, surf_scatters, surf_lines], interval=10, blit=False,
                                   repeat=True)

    f3 = plt.Figure(dpi=100)
    ax3 = f3.add_subplot(111)
    l3 = FigureCanvasTkAgg(f3, root)
    l3.get_tk_widget().grid(row=1, column=0)
    cost_plot(cost, function, ax3)

    f4 = plt.Figure(dpi=100)
    l4 = FigureCanvasTkAgg(f4, root)
    l4.get_tk_widget().grid(row=1, column=1)
    root.mainloop()


def function_of(x, y, function, a=0, b=1, A=10, dimensions=2):
    if function == "rastrigin":  # rastrigin
        return A * dimensions + (x ** 2 - A * np.cos(math.pi * 2 * x)) + (y ** 2 - A * np.cos(math.pi * 2 * y))
    elif function == "rosenbrock":  # rosenbrock a=0,b=1
        return b * (y - x ** 2) ** 2 + (a - x) ** 2


def contour_plot(data, function, ax):
    global fig, scatters, ani
    x = np.arange(np.min(bounds), np.max(bounds) + 0.05, 0.05)
    y = np.arange(np.min(bounds), np.max(bounds) + 0.05, 0.05)

    X, Y = np.meshgrid(x, y)
    zs = np.array(function_of(np.ravel(X), np.ravel(Y), function))
    Z = zs.reshape(X.shape)

    ax.contour(X, Y, Z, levels=50, cmap='viridis')
    ax.title.set_text("2D Contour Plot of Objective Function")

    xs = data[:, :, 0]
    ys = data[:, :, 1]
    scatters = ax.scatter(xs[0], ys[0], c="red", marker="o", vmin=0, vmax=data.shape[1], edgecolors="Black")
    lines = []
    for i in range(data.shape[1]):
        line = ax.plot(xs[0, i], ys[0, i], c="Black", alpha=0.3)
        lines.append(line)
    return data, scatters, lines


def animate_contour(i, data, scatters, lines):
    plot_data = data[i, :, :2]
    scatters.set_offsets(plot_data)
    if i > 5:
        for lnum, line in enumerate(lines):
            xs = data[i - 5:i, lnum, :2]
            line[0].set_data(xs[:, 0], xs[:, 1])


def surface_plot(data, function, ax):
    global fig, scatters, ani2
    x = np.arange(np.min(bounds), np.max(bounds), 0.05)
    y = np.arange(np.min(bounds), np.max(bounds), 0.05)

    X, Y = np.meshgrid(x, y)
    zs = np.array(function_of(np.ravel(X), np.ravel(Y), function))
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap=cm.nipy_spectral, alpha=0.6)
    # ax.contour3D(X, Y, Z, 50, cmap='gray', linestyles="solid")
    ax.title.set_text("3D Plot of Objective Function")

    xs = data[:, :, 0]
    ys = data[:, :, 1]
    zzs = function_of(xs, ys, function)
    scatters = ax.scatter(xs[0], ys[0], zzs[0], c="red", marker="o", vmin=0, vmax=data.shape[1])
    lines = []
    for i in range(data.shape[1]):
        line = ax.plot(xs[0, i], ys[0, i], zzs[0, i], c="red", alpha=0.3)
        lines.append(line)
    return data, zzs, scatters, lines


# Don't mind this too much, I was trying stuff out
def animate_surface(i, data, z_data, scatters, lines):
    plot_data_x = data[i, :, 0]
    plot_data_y = data[i, :, 1]
    plot_data_z = z_data[i]
    scatters._offsets3d = (plot_data_x, plot_data_y, plot_data_z)
    if i > 5:
        for lnum, line in enumerate(lines):
            xs = data[i - 5:i, lnum, :]
            line[0].set_data(xs[:, 0], xs[:, 1])


def cost_plot(data, function, ax):
    ax.set(xlim=[0, data.shape[0]], xlabel='Iterations', ylabel='Best Cost')
    ax.set(ylim=[np.min(data), np.max(data)])
    ax.title.set_text('Min Cost Function')
    ax.plot(data, lw=3)


# ani3 = animation.FuncAnimation(fig, animate3, frames=iterations, fargs=[],interval=100, blit=False, repeat=True)

if __name__ == "__main__":
    function = "rosenbrock"
    data, cost = pso.pso(function=function, optimize_a=True, a=0.9, b=2.0, c=2.0)
    data2, cost2 = gd.gradient_descent(function=function)
    # gui(np.asarray(data), cost, function)
    gui(np.asarray(data2), cost2, function)