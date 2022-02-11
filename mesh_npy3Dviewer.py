try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkfont
from tkinter.filedialog import askopenfilename

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.tri as mtri
from matplotlib import cm
import matplotlib.colors as colors

import scipy.ndimage
from skimage import morphology
from skimage import measure

import numpy as np

import argparse
import csv
import os
import glob

# abdomen
# soft tissues W:400 L:50
CT_WindowWidth = 400.0
CT_WidnowLevel = 50.0

HU_upperbound = CT_WidnowLevel + ( CT_WindowWidth / 2.0 )
HU_lowerbound = CT_WidnowLevel - ( CT_WindowWidth / 2.0 )

x_pixel_length = 0.405 
y_pixel_length = 0.405
z_pixel_length = 1.0
spacing = np.array([    y_pixel_length, 
                        x_pixel_length,
                        z_pixel_length   ])

from PIL import Image

API_Description = """
***** Radiomics Analysis Platform  *****
API Name: npy 3D viewer
Version:    1.0
Developer: Pei-Yan Li
Email:     d05548014@ntu.edu.tw
****************************************
"""

import sys
if os.name == "nt":
    from ctypes import windll, pointer, wintypes
    try:
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass  # this will fail on Windows Server and maybe early Windows


class Model():

    def __init__(self, file_name=None):

        self.data = []
        if file_name is None:
            # Define unit cube.
            vertices = [[0,0,0], [1,0,0], [1,1,0], [0,1,0],
                        [0,0,1], [1,0,1], [1,1,1], [0,1,1]]
            faces = [[1,2,3,4], [1,2,6,5], [2,3,7,6], [3,4,8,7], [4,1,5,8], [5,6,7,8]]
            data = Mesh(vertices, faces)

            self.data = [data]
        else:
            self.load_file(file_name)

    def clear(self):
        self.data = []

    def load_file(self, file_name):

        if file_name.lower().endswith(('.npy')):
            self.load_npy(file_name)

    def load_npy(self, file_name):

        vertices = []
        faces = []

        imgs_to_process = np.load(file_name).astype(np.float64)
        print(f'Shape before resampling: {imgs_to_process.shape}')
        imgs_after_resamp, new_space = self.resample(image=imgs_to_process, new_spacing=[1.0,1.0,0.5]) 
        print(f'Shape after resampling: {imgs_after_resamp.shape}')
        vertices, faces, norm, val = self.make_mesh(imgs_after_resamp, step_size=1)
        num_faces, _ = faces.shape
        num_val, = val.shape
        
        new_val = []
        for nodes in faces:
            new_val.append( (val[nodes[0]] + val[nodes[1]] + val[nodes[2]]) / 3 )
        
        new_val = np.array(new_val)
        norm = colors.Normalize(new_val.min(), new_val.max())
        
        c = cm.gray(norm(new_val))
        mesh = Poly3DCollection(vertices[faces], linewidths=0.05, alpha=1, 
                                facecolor=c)

        hu_val = HU_lowerbound + ( new_val * CT_WindowWidth ) / 255 
        m = cm.ScalarMappable(cmap=cm.gray)
        m.set_array([min(hu_val),max(hu_val)])
        m.set_clim(vmin=min(hu_val),vmax=max(hu_val))

        self.x, self.y, self.z = zip(*vertices)
        self.verts = vertices
        self.faces = faces
        self.norms = norm
        self.vals   = val
        self.mesh = mesh
        self.m = m
        self.file_name = file_name

    def resample(self, image, new_spacing=[1.0, 1.0, 1.0]):

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        
        new_image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
        
        return new_image, new_spacing

    def make_mesh(self, image, step_size=1):
       
        print("Calculating surface")
        level = 0.5 * (image.min() + image.max())
        verts, faces, norm, val = measure.marching_cubes_lewiner(image, level=level,
                                                                 step_size=step_size, 
                                                                 allow_degenerate=True)
        return verts, faces, norm, val

    def get_bounding_box(self):
        bbox = self.data[0].bounding_box
        for mesh in self.data[1:]:
            for i in range(len(bbox)):
                x_i = mesh.bounding_box[i]
                bbox[i][0] = min([bbox[i][0], min(x_i)])
                bbox[i][1] = max([bbox[i][1], max(x_i)])

        return bbox

class Mesh():

    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.bounding_box = self.get_bounding_box()

    def get_vertices(self):
        vertices = []
        for face in self.faces:
            vertices.append([self.vertices[ivt-1] for ivt in face])

        return vertices

    def get_line_segments(self):
        line_segments = set()
        for face in self.faces:
            for i in range(len(face)):
                iv = face[i]
                jv = face[(i+1)%len(face)]
                if jv > iv:
                    edge = (iv, jv)
                else:
                    edge = (jv, iv)

                line_segments.add(edge)

        return [[self.vertices[edge[0]-1], self.vertices[edge[1]-1]] for edge in line_segments]

    def get_bounding_box(self):
        v = [vti for face in self.get_vertices() for vti in face]
        bbox = []
        for i in range(len(self.vertices[0])):
            x_i = [p[i] for p in v]
            bbox.append([min(x_i), max(x_i)])

        return bbox

class View():

    def __init__(self, model=None):

        if model is None:
            model = Model()
        self.model = model

        figure = Figure(tight_layout=True, figsize=(2,1))
        axes = mplot3d.Axes3D(figure, elev=45, 
                                      azim=-175)

        self.figure = figure
        self.axes = axes
        self.canvas = None
        self.toolbar = None
        self.cb = None

        self.plot()

    def clear(self):
        if self.cb != None:
            self.cb.remove()
        self.axes.clear()
        self.update()

    def update(self):
        if self.canvas is not None:
            self.canvas.draw()

    def plot_npy(self):

        self.clear()     
        #self.cb = self.figure.colorbar(self.axes.plot_surface(self.model.x, self.model.y, self.model.z, cmap=cm.jet), shrink=0.5, location='right')  
        
        self.axes.add_collection3d(self.model.mesh)
        self.cb = self.figure.colorbar(self.model.m, shrink=0.5, location='left')

        self.axes.set_xlim(0, max(self.model.x))
        self.axes.set_ylim(0, max(self.model.y))
        self.axes.set_zlim(0, max(self.model.z))

        self.axes.set(zlabel="(mm)", 
                      xlabel="(mm)", 
                      ylabel="(mm)")
        
        self.update()

    def plot(self, types="solid + wireframe"):
        self.clear()
        if isinstance(types, (str,)):
            types = [s.strip() for s in types.split('+')]

        for mesh in self.model.data:
            for type in types:

                if type=="solid":
                    self.axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.get_vertices()))

                elif type=="wireframe":
                    self.axes.add_collection3d(mplot3d.art3d.Line3DCollection(mesh.get_line_segments(),
                                                                              colors=(0.1, 0.1, 0.35, 1)))
                else:
                    # Unknown plot type
                    return None

        if len(self.model.data) >= 1:
            self.axes.auto_scale_xyz(*self.model.get_bounding_box())
            self.update()

    def reset(self):
        self.axes.view_init()
        self.update()

class Controller():

    def __init__(self, view=None, view_2=None):

        root = tk.Tk()
        root.title("3D appearance Viewer")

        if view is None:
            view = View()

        if view_2 is None:
            view_2 = View()

        f1 = ttk.Frame(root)
        f1.pack(side=tk.TOP, anchor=tk.W)

        toolbar = [ tk.Button(f1, text="(1) Open"),
                    tk.Button(f1, text="(1) Reset", command=view.reset),
                    tk.Button(f1, text="(2) Open"),
                    tk.Button(f1, text="(2) Reset", command=view_2.reset) ]
        
        var = tk.StringVar()
        var2 = tk.StringVar()

        toolbar[0].config(command=lambda: self.open(var))
        toolbar[2].config(command=lambda: self.open2(var2))
        [obj.pack(side=tk.LEFT, anchor=tk.W) for obj in toolbar]

        canvas = FigureCanvasTkAgg(view.figure, root)
        canvas.mpl_connect('button_press_event', view.axes._button_press)
        canvas.mpl_connect('button_release_event', view.axes._button_release)
        canvas.mpl_connect('motion_notify_event', view.axes._on_move)
        canvas.draw()
        canvas._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas2 = FigureCanvasTkAgg(view_2.figure, root)
        canvas2.mpl_connect('button_press_event', view_2.axes._button_press)
        canvas2.mpl_connect('button_release_event', view_2.axes._button_release)
        canvas2.mpl_connect('motion_notify_event', view_2.axes._on_move)
        canvas2.draw()
        canvas2._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.root = root
        view.canvas = canvas
        self.view = view
        self.model = view.model

        view_2.canvas = canvas2
        self.view_2 = view_2
        self.model2 = view_2.model

    def render(self):
        self.root.mainloop()

    def open(self, var):
        file_name = askopenfilename( title = "Select file to open",
                                     filetypes = (  ("all files","*.*"),
                                                    ("NPY files","*.npy") ))

        self.model.clear()
        self.model.load_file(file_name)

        if file_name.lower().endswith('.npy'):

            self.view.plot_npy()

        else:

            self.view.plot(var.get())   

    def open2(self, var2):
        file_name = askopenfilename( title = "Select file to open",
                                     filetypes = (  ("all files","*.*"),
                                                    ("NPY files","*.npy") ))
        self.model2.clear()
        self.model2.load_file(file_name)

        if file_name.lower().endswith('.npy'):

            self.view_2.plot_npy()
        
        else:
            self.view_2.plot(var2.get())   

    def exit(self):
        self.model.clear()
        self.view.clear()
        self.model2.clear()
        self.view_2.clear()
        self.root.destroy()

def setMaxWidth(stringList, element):
    try:
        f = tkfont.nametofont(element.cget("font"))
        zerowidth = f.measure("0")
    except:
        f = tkfont.nametofont(ttk.Style().lookup("TButton", "font"))
        zerowidth = f.measure("0") - 0.8

    w = max([f.measure(i) for i in stringList])/zerowidth
    element.config(width=int(w))

class App():

    def __init__(self, model1=None, view1=None, 
                       model2=None, view2=None, 
                 controller=None):
        file_name1 = None
        file_name2 = None
        if len(sys.argv) >= 2:
            file_name1 = sys.argv[1]

        if len(sys.argv) >= 3:
            file_name2 = sys.argv[2]

        if model1 is None:
            model1 = Model(file_name1)

        if model2 is None:
            model2 = Model(file_name2)

        if view1 is None:
            view1 = View(model1)

        if view2 is None:
            view2 = View(model2)

        if controller is None:
            controller = Controller(view=view1, view_2=view2)

        self.model1 = model1
        self.view1 = view1
        self.model2 = model2
        self.view2 = view2
        self.controller = controller

    def start(self):
        self.controller.render()

if __name__ == "__main__":

    app = App()
    app.start()
    