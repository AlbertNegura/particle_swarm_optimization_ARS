import math

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import style, cm

import tkinter as tk
from tkinter import *

import numpy as np
import particle_swarm_optimization

LARGE_FONT= ("Verdana", 12)
style.use("ggplot")

class PSO(tk.Tk):
    frames = {}
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Particle Swarm Optimization")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        f = 0

        for F in (StartPage, VisualizationPage):
            if f == 0:
                frame = F(container, self)
                f+=1
            else:
                frame = F(container, self, StartPage)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):
    omega = 0.9
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label1 = tk.Label(self, text=("""Particle Swarm Optimization Visualization\nAuthors: Julien Havel, Albert Negura, Sergi Nogues"""), font=LARGE_FONT)
        label1.pack(pady=10,padx=10)

        self.omega_slider = tk.Scale(self, from_=0.00, to=1.00, length=300,tickinterval=10, digits=3, resolution=0.01, orient=HORIZONTAL, label="Omega")
        self.omega_slider.set(0.90)
        self.omega_slider.pack()
        self.social_slider = tk.Scale(self, from_=0.00, to=10.00, length=300,tickinterval=10, digits=4, resolution=0.01, orient=HORIZONTAL, label="Social Constant")
        self.social_slider.set(2.00)
        self.social_slider.pack()
        self.cognitive_slider = tk.Scale(self, from_=0.00, to=10.00, length=300,tickinterval=10, digits=4, resolution=0.01, orient=HORIZONTAL, label="Cognitive Constant")
        self.cognitive_slider.set(2.00)
        self.cognitive_slider.pack()
        self.population_slider = tk.Scale(self, from_=0, to=100, length=300,tickinterval=10, orient=HORIZONTAL, label="Population")
        self.population_slider.set(20)
        self.population_slider.pack()
        self.iterations_slider = tk.Scale(self, from_=0, to=1000, length=300,tickinterval=100, orient=HORIZONTAL, label="Iterations")
        self.iterations_slider.set(50)
        self.iterations_slider.pack()


        label2 = tk.Label(self, text=("""Select function to minimize."""), font=LARGE_FONT)
        label2.pack(pady=10,padx=10)
        self.function = "rosenbrock"
        self.function_radio1 = tk.Radiobutton(self, text="Rosenbrock (DEFAULT)", variable=self.function, value="rosenbrock")
        self.function_radio1.pack()
        self.function_radio2 = tk.Radiobutton(self, text="Rastrigin", variable=self.function, value="rastrigin")
        self.function_radio2.pack()


        label3 = tk.Label(self, text=("""Select algorithm (note that gradient descent is independent of sliders and neighbourhood selection)."""), font=LARGE_FONT)
        label3.pack(pady=10,padx=10)
        self.algorithm = "pso"
        self.algorithm_radio1 = tk.Radiobutton(self, text="Particle Swarm Optimization (DEFAULT)", variable=self.algorithm, value="pso")
        self.algorithm_radio1.pack()
        self.algorithm_radio2 = tk.Radiobutton(self, text="Gradient Descent (note that sliders don't do anything)", variable=self.algorithm, value="gd")
        self.algorithm_radio2.pack()

        label4 = tk.Label(self, text=("""Select neighbourhood behaviour for PSO."""), font=LARGE_FONT)
        label4.pack(pady=10,padx=10)
        self.neighbourhood = "global"
        self.neighbourhood_radio1 = tk.Radiobutton(self, text="Global Neighbourhood (DEFAULT)", variable=self.neighbourhood, value="global")
        self.neighbourhood_radio1.pack()
        self.neighbourhood_radio2 = tk.Radiobutton(self, text="Social Neighbourhood with 2 Neighbours", variable=self.neighbourhood, value="social-two")
        self.neighbourhood_radio2.pack()
        self.neighbourhood_radio3 = tk.Radiobutton(self, text="Social Neighbourhood with 4 Neighbours", variable=self.neighbourhood, value="social-four")
        self.neighbourhood_radio3.pack()
        self.neighbourhood_radio4 = tk.Radiobutton(self, text="Geographical Neighbourhood with 2 Nearest Neighbours", variable=self.neighbourhood, value="geographical")
        self.neighbourhood_radio4.pack()

        button1 = Button(self, text="Visualize",
                            command=lambda: self.showFrame(parent=parent,controller=controller))
        button1.pack()

        button2 = Button(self, text="Exit",
                            command=quit)
        button2.pack()

    def showFrame(self,parent,controller):
        self.update_all()
        controller.show_frame(VisualizationPage)

    def update_all(self):
        self.omega = self.omega_slider.get()
        self.social = self.social_slider.get()
        self.cognitive = self.cognitive_slider.get()
        self.population = self.population_slider.get()
        self.iterations = self.iterations_slider.get()
        self.neighbourhood = self.neighbourhood
        self.function = self.function
        self.algorithm = self.algorithm

class VisualizationPage(tk.Frame):

    def __init__(self, parent, controller, home):
        self.controller = controller
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Visualization", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        self.figure = Figure(figsize=(10,6), dpi=100)
        self.ax1=self.figure.add_subplot(221)
        self.ax2=self.figure.add_subplot(222, projection="3d")
        self.ax3=self.figure.add_subplot(223)
        self.ax4=self.figure.add_subplot(224)
        self.canvas = FigureCanvasTkAgg(self.figure, self)

        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        button2 = Button(self, text="Execute",
                            command=lambda: self.execute())
        button2.pack()

        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)


    def function_of(self, x, y, a=0, b=1, A=10, dimensions=2):
        if self.function == "rastrigin":  # rastrigin
            return A * dimensions + (x ** 2 - A * np.cos(math.pi * 2 * x)) + (y ** 2 - A * np.cos(math.pi * 2 * y))
        elif self.function == "rosenbrock":  # rosenbrock a=0,b=1
            return b * (y - x ** 2) ** 2 + (0 - x) ** 2

    def contour_plot(self, data):
        x = np.arange(np.min(self.bounds), np.max(self.bounds) + 0.05, 0.05)
        y = np.arange(np.min(self.bounds), np.max(self.bounds) + 0.05, 0.05)
        X, Y = np.meshgrid(x, y)
        zs = np.array(self.function_of(np.ravel(X), np.ravel(Y)))
        Z = zs.reshape(X.shape)


        self.ax1.contour(X, Y, Z, levels=50, cmap='viridis')
        self.ax1.title.set_text("2D Contour Plot of Objective Function")

        xs = data[:, :, 0]
        ys = data[:, :, 1]
        scatters = self.ax1.scatter(xs[0], ys[0], c="red", marker="o", vmin=0, vmax=data.shape[1], edgecolors="Black")
        lines = []
        for i in range(data.shape[1]):
            line = self.ax1.plot(xs[0, i], ys[0, i], c="Black", alpha=0.3)
            lines.append(line)
        return data, scatters, lines

    def animate_contour(self, i, data, scatters, lines):
        plot_data = data[i, :, :2]
        scatters.set_offsets(plot_data)
        if i > 0:
            for lnum, line in enumerate(lines):
                if i == 2:
                    xs = data[i - 2:i, lnum, :2]
                    line[0].set_data(xs[:, 0], xs[:, 1])
                if i == 3:
                    xs = data[i - 3:i, lnum, :2]
                    line[0].set_data(xs[:, 0], xs[:, 1])
                if i == 4:
                    xs = data[i - 4:i, lnum, :2]
                    line[0].set_data(xs[:, 0], xs[:, 1])
                if i >= 5:
                    xs = data[i - 5:i, lnum, :2]
                    line[0].set_data(xs[:, 0], xs[:, 1])

    def surface_plot(self,data):
        x = np.arange(np.min(self.bounds), np.max(self.bounds) + 0.05, 0.05)
        y = np.arange(np.min(self.bounds), np.max(self.bounds) + 0.05, 0.05)
        X, Y = np.meshgrid(x, y)
        zs = np.array(self.function_of(np.float32(np.ravel(X)), np.float32(np.ravel(Y))))
        Z = zs.reshape(X.shape)

        self.ax2.plot_surface(X, Y, Z, cmap=cm.nipy_spectral, alpha=0.6)
        # ax.contour3D(X, Y, Z, 50, cmap='gray', linestyles="solid")
        self.ax2.title.set_text("3D Plot of Objective Function")

        xs = data[:, :, 0]
        ys = data[:, :, 1]
        zzs = self.function_of(xs, ys, self.function)
        scatters = self.ax2.scatter(xs[0], ys[0], zzs[0], c="red", marker="o", vmin=0, vmax=data.shape[1])
        lines = []
        for i in range(data.shape[1]):
            line = self.ax2.plot(xs[0, i], ys[0, i], zzs[0, i], c="red", alpha=0.3)
            lines.append(line)
        return data, zzs, scatters, lines

    # Don't mind this too much, I was trying stuff out
    def animate_surface(self,i, data, z_data, scatters, lines):
        plot_data_x = data[i, :, 0]
        plot_data_y = data[i, :, 1]
        plot_data_z = z_data[i]
        scatters._offsets3d = (plot_data_x, plot_data_y, plot_data_z)
        if i > 0:
            for lnum, line in enumerate(lines):
                if i == 2:
                    xs = data[i - 2:i, lnum, :2]
                    line[0].set_data(xs[:, 0], xs[:, 1])
                if i == 3:
                    xs = data[i - 3:i, lnum, :2]
                    line[0].set_data(xs[:, 0], xs[:, 1])
                if i == 4:
                    xs = data[i - 4:i, lnum, :2]
                    line[0].set_data(xs[:, 0], xs[:, 1])
                if i >= 5:
                    xs = data[i - 5:i, lnum, :2]
                    line[0].set_data(xs[:, 0], xs[:, 1])

    def cost_plot(self,data):
        self.ax3.set(xlim=[0, data.shape[0]], xlabel='Iterations', ylabel='Best Cost')
        self.ax3.set(ylim=[np.min(data), np.max(data)])
        self.ax3.title.set_text('Min Cost Function')
        self.ax3.plot(data, lw=3)

    def execute(self):
        global ani1, ani2
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()

        PSO.frames[StartPage].update_all()
        self.function = PSO.frames[StartPage].function
        self.population = PSO.frames[StartPage].population
        self.iterations = PSO.frames[StartPage].iterations
        self.bounds = [-2.4, 2.4] if self.function == "rosenbrock" else [-5.12, 5.12]
        data, cost = particle_swarm_optimization.pso(population=PSO.frames[StartPage].population, iterations=PSO.frames[StartPage].iterations, function=self.function, optimize_a=False, a=PSO.frames[StartPage].omega, b=PSO.frames[StartPage].social, c=PSO.frames[StartPage].cognitive)
        data = np.array(data)
        if PSO.frames[StartPage].algorithm == "pso":
            cont_data, cont_scatters, cont_lines = self.contour_plot(data)
            surf_data, surf_zs, surf_scatters, surf_lines = self.surface_plot(data)
            self.cost_plot(cost)
        elif PSO.frames[StartPage].algorithm == "gd":
            pass

        ani1 = animation.FuncAnimation(self.figure, self.animate_contour, frames=self.iterations,
                                       fargs=[cont_data, cont_scatters, cont_lines], interval=10,
                                       blit=False, repeat=True)
        ani2 = animation.FuncAnimation(self.figure, self.animate_surface, frames=self.iterations,
                                       fargs=[surf_data, surf_zs, surf_scatters, surf_lines], interval=10, blit=False,
                                      repeat=True)
        self.refresh()

    def refresh(self):
        self.canvas.draw()




app = PSO()
#ani = animation.FuncAnimation(f, animate, interval=1000)
app.mainloop()