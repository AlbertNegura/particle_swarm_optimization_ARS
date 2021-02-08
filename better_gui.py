import math

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style, cm

import tkinter as tk
from tkinter import *
from tkinter import ttk

import numpy as np
import particle_swarm_optimization
import gradient_descent

LARGE_FONT= ("Verdana", 12)
style.use("seaborn-muted")
ani1 = None

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
        label1 = ttk.Label(self, text=("""Particle Swarm Optimization Visualization\nAuthors: Julien Havel, Albert Negura, Sergi Nogues"""), font=LARGE_FONT)
        label1.pack(pady=10,padx=10)

        self.omega_slider = tk.Scale(self, from_=0.00, to=1.00, length=600,tickinterval=10, digits=3, resolution=0.01, orient=HORIZONTAL, label="Omega")
        self.omega_slider.pack()
        self.social_slider = tk.Scale(self, from_=0.00, to=10.00, length=600,tickinterval=10, digits=4, resolution=0.01, orient=HORIZONTAL, label="Social Constant")
        self.social_slider.pack()
        self.cognitive_slider = tk.Scale(self, from_=0.00, to=10.00, length=600,tickinterval=10, digits=4, resolution=0.01, orient=HORIZONTAL, label="Cognitive Constant")
        self.cognitive_slider.pack()
        self.population_slider = tk.Scale(self, from_=0, to=100, length=600,tickinterval=10, orient=HORIZONTAL, label="Population")
        self.population_slider.pack()
        self.iterations_slider = tk.Scale(self, from_=0, to=1000, length=600,tickinterval=100, orient=HORIZONTAL, label="Iterations")
        self.iterations_slider.pack()


        label2 = ttk.Label(self, text=("""Select function to minimize."""), font=LARGE_FONT)
        label2.pack(pady=10,padx=10)
        self.function = "rosenbrock"
        self.func_var = IntVar(self)
        self.function_radio1 = ttk.Radiobutton(self, text="Rosenbrock (DEFAULT)", variable=self.func_var, value=0, command=self.set_func)
        self.function_radio1.pack()
        self.function_radio2 = ttk.Radiobutton(self, text="Rastrigin", variable=self.func_var, value=1, command=self.set_func)
        self.function_radio2.pack()


        label3 = ttk.Label(self, text=("""Select algorithm (note that gradient descent is independent of sliders and neighbourhood selection)."""), font=LARGE_FONT)
        label3.pack(pady=10,padx=10)
        self.algorithm = "pso"
        self.alg_var = IntVar(self)
        self.algorithm_radio1 = ttk.Radiobutton(self, text="Particle Swarm Optimization (DEFAULT)", variable=self.alg_var, value=0, command=self.set_algo)
        self.algorithm_radio1.pack()
        self.algorithm_radio2 = ttk.Radiobutton(self, text="Gradient Descent (note that sliders don't do anything)", variable=self.alg_var, value=1, command=self.set_algo)
        self.algorithm_radio2.pack()

        label4 = ttk.Label(self, text=("""Select neighbourhood behaviour for PSO."""), font=LARGE_FONT)
        label4.pack(pady=10,padx=10)
        self.neighbourhood = "global"
        self.neigh_var = IntVar(self)
        self.neighbourhood_radio1 = ttk.Radiobutton(self, text="Global Neighbourhood (DEFAULT)", variable=self.neigh_var, value=0, command=self.set_neighbourhood)
        self.neighbourhood_radio1.pack()
        self.neighbourhood_radio2 = ttk.Radiobutton(self, text="Social Neighbourhood with 2 Neighbours", variable=self.neigh_var, value=1, command=self.set_neighbourhood)
        self.neighbourhood_radio2.pack()
        self.neighbourhood_radio3 = ttk.Radiobutton(self, text="Social Neighbourhood with 4 Neighbours", variable=self.neigh_var, value=2, command=self.set_neighbourhood)
        self.neighbourhood_radio3.pack()
        self.neighbourhood_radio4 = ttk.Radiobutton(self, text="Geographical Neighbourhood with 2 Nearest Neighbours", variable=self.neigh_var, value=3, command=self.set_neighbourhood)
        self.neighbourhood_radio4.pack()

        button1 = ttk.Button(self, text="Visualize",
                            command=lambda: self.showFrame(parent=parent,controller=controller))
        button1.pack()

        button2 = ttk.Button(self, text="Reset to Default Values",
                            command=lambda: self.set_default())
        button2.pack()

        button3 = ttk.Button(self, text="Exit",
                            command=quit)
        button3.pack()

        self.update_all()
        self.set_default()
        self.update_idletasks()

    def set_neighbourhood(self):
        self.neighbourhood = "global" if self.neigh_var.get()==0 else "social-two"  if self.neigh_var.get()==1 else "social-four" if self.neigh_var.get()==2  else "geographical"

    def set_algo(self):
        self.algorithm = "pso" if self.alg_var.get()==0 else "gd"

    def set_func(self):
        self.function = "rosenbrock" if self.func_var.get()==0 else "rastrigin"

    def showFrame(self,parent,controller):
        PSO.frames[StartPage].first_run = True
        self.update_all()
        controller.show_frame(VisualizationPage)

    def update_all(self):
        self.omega = self.omega_slider.get()
        self.social = self.social_slider.get()
        self.cognitive = self.cognitive_slider.get()
        self.population = self.population_slider.get()
        self.iterations = self.iterations_slider.get()
        self.neighbourhood = "global" if self.neigh_var.get()==0 else "social-two"  if self.neigh_var.get()==1 else "social-four" if self.neigh_var.get()==2  else "geographical"
        self.function = "rosenbrock" if self.func_var.get()==0 else "rastrigin"
        self.algorithm = "pso" if self.alg_var.get()==0 else "gd"

    def set_default(self):
        self.update_idletasks()
        self.iterations_slider.set(50)
        self.population_slider.set(20)
        self.cognitive_slider.set(2.00)
        self.social_slider.set(2.00)
        self.omega_slider.set(0.90)
        self.update_idletasks()

class VisualizationPage(tk.Frame):

    def __init__(self, parent, controller, home):
        self.controller = controller
        ttk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="Visualization", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: self.go_back(controller))
        button1.pack()

        self.figure = Figure(figsize=(6,6), dpi=100)
        self.ax1=self.figure.add_subplot(221)
        self.ax2=self.figure.add_subplot(222, projection="3d")
        self.ax3=self.figure.add_subplot(223)
        self.ax4=self.figure.add_subplot(224)
        self.canvas = FigureCanvasTkAgg(self.figure, self)

        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        button2 = ttk.Button(self, text="Execute",
                            command=lambda: self.execute())
        button2.pack()
        button3 = ttk.Button(self, text="Step",
                            command=lambda: self.step())
        button3.pack()
        self.iterations_slider = tk.Scale(self, from_=0, to=PSO.frames[StartPage].iterations, length=600,tickinterval=int(PSO.frames[StartPage].iterations/10), orient=HORIZONTAL, label="Next Time Step")
        self.iterations_slider.pack()
        self.first_run = True
        self.minimum = [0,0]
        if PSO.frames[StartPage].algorithm == "pso":
            self.text = ("Algorithm: "+ ("Particle Swarm Optimization" if PSO.frames[StartPage].algorithm else "Gradient Descent") + " on function: " + PSO.frames[StartPage].function + "."+"\nPopulation="+str(PSO.frames[StartPage].population)+";iterations="+str(PSO.frames[StartPage].iterations)+"\nomega="+str(PSO.frames[StartPage].omega)+" social constant="+str(PSO.frames[StartPage].social)+" cognitive constant="+str(PSO.frames[StartPage].cognitive))
        elif PSO.frames[StartPage].algorithm == "gd":
            self.text = ("Algorithm: "+ ("Gradient Descent" if PSO.frames[StartPage].algorithm else "Gradient Descent") + " on function: " + PSO.frames[StartPage].function + ".")

        self.label2 = ttk.Label(self, text=self.text, font=LARGE_FONT)
        self.label2.pack(pady=10,padx=10)

        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)
        self.iterations_slider_shower = tk.Scale(self, from_=0, to=PSO.frames[StartPage].iterations, length=600,tickinterval=int(PSO.frames[StartPage].iterations/10), orient=HORIZONTAL, label="Current Time Step (non-interactable slider)")
        self.iterations_slider_shower.pack()


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
        levels = 50
        if self.function == "rastrigin":
            levels=10
        self.ax1.contourf(X, Y, Z, levels=levels, cmap='viridis',alpha=0.3)
        self.ax1.scatter(0,0, c="white",marker="*", edgecolors="black", s=250)
        self.ax1.title.set_text("2D Contour Plot of Objective Function")

        xs = data[:, :, 0]
        ys = data[:, :, 1]
        scatters = self.ax1.scatter(xs[0], ys[0], c="red", marker="o", vmin=0, vmax=data.shape[1], edgecolors="Black")
        lines = []
        for i in range(data.shape[1]):
            line = self.ax1.plot(xs[0, i], ys[0, i], c="Black", alpha=0.6)
            lines.append(line)
        return data, scatters, lines

    def contour_plot_step(self, data, time):
        x = np.arange(np.min(self.bounds), np.max(self.bounds) + 0.05, 0.05)
        y = np.arange(np.min(self.bounds), np.max(self.bounds) + 0.05, 0.05)
        X, Y = np.meshgrid(x, y)
        zs = np.array(self.function_of(np.ravel(X), np.ravel(Y)))
        Z = zs.reshape(X.shape)

        levels = 50
        if self.function == "rastrigin":
            levels=10

        self.ax1.contourf(X, Y, Z, levels=levels, cmap='viridis',alpha=0.3)
        self.ax1.scatter(0,0, c="white",marker="*", edgecolors="black", s=250)
        self.ax1.title.set_text("2D Contour Plot of Objective Function")
        xs = data[time, :, 0]
        ys = data[time, :, 1]
        scatters = self.ax1.scatter(xs, ys, c="red", marker="o", vmin=0, vmax=data.shape[1], edgecolors="Black")
        lines = []
        for i in range(data.shape[1]):
            line = self.ax1.plot(data[:time, i,0], data[:time, i,1], alpha=0.6)
            lines.append(line)
        return data, scatters, lines


    def surface_plot(self,data):
        x = np.arange(np.min(self.bounds), np.max(self.bounds) + 0.05, 0.05)
        y = np.arange(np.min(self.bounds), np.max(self.bounds) + 0.05, 0.05)
        X, Y = np.meshgrid(x, y)
        zs = np.array(self.function_of(np.float32(np.ravel(X)), np.float32(np.ravel(Y))))
        Z = zs.reshape(X.shape)
        if self.function == "rastrigin":
            self.ax2.plot_wireframe(X, Y, Z, cmap="viridis", alpha=0.15, rstride=10, cstride=10)
            self.ax2.contour(X, Y, Z, cmap="viridis", alpha=0.25, levels=10)
        else:
            self.ax2.plot_wireframe(X, Y, Z, cmap="viridis", alpha=0.65, rstride=10, cstride=10)
            self.ax2.contour(X, Y, Z, cmap="viridis", alpha=0.55, levels=50)
        self.ax2.view_init(elev=66., azim=50)
        self.ax2.scatter(0,0,0, c="white",marker="*", edgecolors="black", s=250)
        # ax.contour3D(X, Y, Z, 50, cmap='gray', linestyles="solid")
        self.ax2.title.set_text("3D Plot of Objective Function")

        xs = data[:, :, 0]
        ys = data[:, :, 1]
        zzs = self.function_of(xs, ys, self.function)
        scatters = self.ax2.scatter(xs[0], ys[0], zzs[0]+0.01, s=4, c="red", marker="o", vmin=0, vmax=data.shape[1], edgecolors="Black")
        lines = []
        for i in range(data.shape[1]):
            line = self.ax2.plot(xs[0, i], ys[0, i], zzs[0, i], c="red", alpha=0.3)
            lines.append(line)
        return data, zzs, scatters, lines

    def surface_plot_step(self,data, time):
        x = np.arange(np.min(self.bounds), np.max(self.bounds) + 0.05, 0.05)
        y = np.arange(np.min(self.bounds), np.max(self.bounds) + 0.05, 0.05)
        X, Y = np.meshgrid(x, y)
        zs = np.array(self.function_of(np.float32(np.ravel(X)), np.float32(np.ravel(Y))))
        Z = zs.reshape(X.shape)
        self.ax2.plot_wireframe(X, Y, Z, cmap="viridis", alpha=0.25, rstride=10, cstride=10)
        self.ax2.view_init(elev=66., azim=50)
        self.ax2.scatter(0,0,0, c="white",marker="*", edgecolors="black", s=250)
        # ax.contour3D(X, Y, Z, 50, cmap='gray', linestyles="solid")
        self.ax2.title.set_text("3D Plot of Objective Function")

        xs = data[time, :, 0]
        ys = data[time, :, 1]
        zzs = self.function_of(xs, ys, self.function)
        scatters = self.ax2.scatter(xs, ys, zzs+0.01, s=4, c="red", marker="o", vmin=0, vmax=data.shape[1], edgecolors="Black")
        lines = []
        return data, zzs, scatters, lines

    def cost_plot(self,data):
        self.ax3.set(xlim=[0, data.shape[0]], xlabel='Iterations', ylabel='Best Cost')
        self.ax3.set(ylim=[0, np.max(data)])
        self.ax3.title.set_text('Min Cost Function with Min Cost = {:.3f}'.format((np.min(data))))
        self.ax3.plot(data, lw=3)

    def av_cost_plot(self,data):
        self.ax4.set(xlim=[0, data.shape[0]], xlabel='Iterations', ylabel='Best Cost')
        self.ax4.set(ylim=[0, np.max(data)])
        self.ax4.title.set_text('Average Cost Function')
        self.ax4.plot(data, lw=3)

    def animate(self, i, data, scatters, lines, surf_data, surf_zs, surf_scatters, surf_lines, algorithm):
        self.iterations_slider_shower.set(int(i))
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
        plot_data_x = surf_data[i, :, 0]
        plot_data_y = surf_data[i, :, 1]
        plot_data_z = surf_zs[i]
        surf_scatters._offsets3d = (plot_data_x, plot_data_y, plot_data_z + 0.1)
        if i > 0:
            for lnum, line in enumerate(surf_lines):
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

    def execute(self):
        global ani1
        if ani1 is not None:
            ani1.event_source.stop()
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        PSO.frames[StartPage].update_all()
        self.function = PSO.frames[StartPage].function
        self.population = PSO.frames[StartPage].population
        self.iterations = PSO.frames[StartPage].iterations
        if self.iterations > 0:
            self.iterations_slider_shower.configure(to=self.iterations-1)
        self.bounds = [-2.4, 2.4] if self.function == "rosenbrock" else [-5.12, 5.12]
        data, cost, av_cost = particle_swarm_optimization.pso(population=PSO.frames[StartPage].population, iterations=PSO.frames[StartPage].iterations, function=self.function, optimize_a=False, a=PSO.frames[StartPage].omega, b=PSO.frames[StartPage].social, c=PSO.frames[StartPage].cognitive)
        data = np.array(data)
        if PSO.frames[StartPage].algorithm == "pso":
            cont_data, cont_scatters, cont_lines = self.contour_plot(data)
            surf_data, surf_zs, surf_scatters, surf_lines = self.surface_plot(data)
            self.cost_plot(cost)
            self.av_cost_plot(av_cost)
        elif PSO.frames[StartPage].algorithm == "gd":
            data2, cost2 = gradient_descent.gradient_descent(function=self.function)
            data2 = np.array(data2)
            cont_data, cont_scatters, cont_lines = self.contour_plot(data2)
            surf_data, surf_zs, surf_scatters, surf_lines = self.surface_plot(data2)
            self.cost_plot(cost2)
        else: #default to PSO
            cont_data, cont_scatters, cont_lines = self.contour_plot(data)
            surf_data, surf_zs, surf_scatters, surf_lines = self.surface_plot(data)
            self.cost_plot(cost)
            self.av_cost_plot(av_cost)

        ani1 = animation.FuncAnimation(self.figure, self.animate, frames=self.iterations,
                                           fargs=[cont_data, cont_scatters, cont_lines, surf_data, surf_zs, surf_scatters, surf_lines, PSO.frames[StartPage].algorithm], interval=10,
                                           blit=False, repeat=True)
        if PSO.frames[StartPage].algorithm == "pso":
            self.text = ("Algorithm: "+ ("Particle Swarm Optimization" if PSO.frames[StartPage].algorithm else "Gradient Descent") + " on function: " + PSO.frames[StartPage].function + "."+"\nPopulation="+str(PSO.frames[StartPage].population)+";iterations="+str(PSO.frames[StartPage].iterations)+"\nomega="+str(PSO.frames[StartPage].omega)+" social constant="+str(PSO.frames[StartPage].social)+" cognitive constant="+str(PSO.frames[StartPage].cognitive))
            self.label2.config(text=self.text)
        elif PSO.frames[StartPage].algorithm == "gd":
            self.text = ("Algorithm: "+ ("Gradient Descent" if PSO.frames[StartPage].algorithm else "Gradient Descent") + " on function: " + PSO.frames[StartPage].function + ".")
            self.label2.config(text=self.text)

        self.first_run = True
        self.refresh()
        self.update_idletasks()

    def step(self):
        global data, cost, av_cost, i, ani
        if ani1 is not None:
            ani1.event_source.stop()
        if self.first_run:
            i=0
            self.first_run = False
            self.ax1.cla()
            self.ax2.cla()
            self.ax3.cla()
            self.ax4.cla()
            PSO.frames[StartPage].update_all()
            self.function = PSO.frames[StartPage].function
            self.population = PSO.frames[StartPage].population
            self.iterations = PSO.frames[StartPage].iterations
            if self.iterations > 0:
                self.iterations_slider.configure(to=self.iterations-1)
                self.iterations_slider_shower.configure(to=self.iterations-1)
            self.iterations_slider_shower.set(int(i))
            self.bounds = [-2.4, 2.4] if self.function == "rosenbrock" else [-5.12, 5.12]
            if PSO.frames[StartPage].algorithm == "pso":
                data, cost, av_cost = particle_swarm_optimization.pso(population=PSO.frames[StartPage].population, iterations=PSO.frames[StartPage].iterations, function=self.function, optimize_a=False, a=PSO.frames[StartPage].omega, b=PSO.frames[StartPage].social, c=PSO.frames[StartPage].cognitive)
                data = np.array(data)
                cont_data, cont_scatters, cont_lines = self.contour_plot_step(data,i)
                surf_data, surf_zs, surf_scatters, surf_lines = self.surface_plot_step(data,i)
                self.cost_plot(cost)
                self.av_cost_plot(av_cost)
            elif PSO.frames[StartPage].algorithm == "gd":
                data, cost = gradient_descent.gradient_descent(function=self.function)
                data = np.array(data)
                cont_data, cont_scatters, cont_lines = self.contour_plot_step(data,i)
                surf_data, surf_zs, surf_scatters, surf_lines = self.surface_plot_step(data,i)
                self.cost_plot(cost)
            else: #default to PSO
                data, cost, av_cost = particle_swarm_optimization.pso(population=PSO.frames[StartPage].population, iterations=PSO.frames[StartPage].iterations, function=self.function, optimize_a=False, a=PSO.frames[StartPage].omega, b=PSO.frames[StartPage].social, c=PSO.frames[StartPage].cognitive)
                data = np.array(data)
                cont_data, cont_scatters, cont_lines = self.contour_plot_step(data,i)
                surf_data, surf_zs, surf_scatters, surf_lines = self.surface_plot_step(data,i)
                self.cost_plot(cost)
                self.av_cost_plot(av_cost)

            if PSO.frames[StartPage].algorithm == "pso":
                self.text = ("Algorithm: "+ ("Particle Swarm Optimization" if PSO.frames[StartPage].algorithm else "Gradient Descent") + " on function: " + PSO.frames[StartPage].function + "."+"\nPopulation="+str(PSO.frames[StartPage].population)+";iterations="+str(PSO.frames[StartPage].iterations)+"\nomega="+str(PSO.frames[StartPage].omega)+" social constant="+str(PSO.frames[StartPage].social)+" cognitive constant="+str(PSO.frames[StartPage].cognitive))
                self.label2.config(text=self.text)
            elif PSO.frames[StartPage].algorithm == "gd":
                self.text = ("Algorithm: "+ ("Gradient Descent" if PSO.frames[StartPage].algorithm else "Gradient Descent") + " on function: " + PSO.frames[StartPage].function + ".")
                self.label2.config(text=self.text)
            i+=1
            self.iterations_slider.set(i)
        else:
            if(i != self.iterations_slider.get()):
                i = self.iterations_slider.get()
            if(i >= self.iterations-1):
                i = 0
                self.iterations_slider.set(0)
            self.iterations_slider_shower.set(int(i))
            self.ax1.cla()
            self.ax2.cla()
            self.ax3.cla()
            self.ax4.cla()
            cont_data, cont_scatters, cont_lines = self.contour_plot_step(data,i)
            surf_data, surf_zs, surf_scatters, surf_lines = self.surface_plot_step(data,i)
            self.cost_plot(cost)
            self.av_cost_plot(av_cost)
            i+=1
            self.iterations_slider.set(i)

        self.refresh()
        self.update_idletasks()
    def refresh(self):
        self.canvas.draw()

    def go_back(self, controller):
        if ani1 is not None:
            ani1.event_source.stop()
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        controller.show_frame(StartPage)



app = PSO()
app.mainloop()