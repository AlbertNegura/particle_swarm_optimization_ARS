import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style

import tkinter as tk
from tkinter import *

import numpy as np
import particle_swarm_optimization

LARGE_FONT= ("Verdana", 12)
style.use("ggplot")

f = Figure(figsize=(10,6), dpi=100)
a1 = f.add_subplot(221)
a2 = f.add_subplot(222, projection="3d")
a3 = f.add_subplot(223)
a4 = f.add_subplot(224)



def animate(i):
    xList = []
    yList = []

    a1.clear()
    a1.plot(xList, yList)

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
        label1 = tk.Label(self, text=("""Particle Swarm Optimization Visualization - Please set initial parameters!"""), font=LARGE_FONT)
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
        self.function_radio1 = tk.Radiobutton(self, text="Rosenbrock", variable=self.function, value="rosenbrock")
        self.function_radio1.pack()
        self.function_radio2 = tk.Radiobutton(self, text="Rastrigin", variable=self.function, value="rastrigin")
        self.function_radio2.pack()


        label3 = tk.Label(self, text=("""Select algorithm (note that gradient descent is independent of sliders and neighbourhood selection)."""), font=LARGE_FONT)
        label3.pack(pady=10,padx=10)
        self.algorithm = "pso"
        self.algorithm_radio1 = tk.Radiobutton(self, text="Particle Swarm Optimization", variable=self.algorithm, value="pso")
        self.algorithm_radio1.pack()
        self.algorithm_radio2 = tk.Radiobutton(self, text="Gradient Descent (note that sliders don't do anything)", variable=self.algorithm, value="gd")
        self.algorithm_radio2.pack()

        label4 = tk.Label(self, text=("""Select neighbourhood behaviour for PSO."""), font=LARGE_FONT)
        label4.pack(pady=10,padx=10)
        self.neighbourhood = "global"
        self.neighbourhood_radio1 = tk.Radiobutton(self, text="Global Neighbourhood", variable=self.neighbourhood, value="global")
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
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Visualization", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()


        canvas = FigureCanvasTkAgg(f, self)
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        print(home.omega)




app = PSO()
ani = animation.FuncAnimation(f, animate, interval=1000)
app.mainloop()