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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.tri as mtri
from matplotlib import cm

import numpy as np

import sys
import os
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
        '''Load mesh from file
        '''
        if file_name.lower().endswith(('.stl','.stla','.stlb')):
            self.load_stl(file_name)

        elif file_name.lower().endswith('.obj'):
            self.load_obj(file_name)
        
        elif file_name.lower().endswith('.tif'):
            self.load_tif(file_name)

        elif file_name.lower().endswith('.png'):
            self.load_png(file_name)

    def load_png(self, file_name):

        vertices = []
        faces = []

        img = mpimg.imread(file_name)
        img_height, img_width = img.shape

        X, Y = np.meshgrid(range(img_width), range(img_height));
        tri = mtri.Triangulation(X.flatten(), Y.flatten())

        width       = 400
        lower_bound = 50 - (width / 2.0)

        self.x = X
        self.y = Y
        self.z = img * (width / 255) + lower_bound  
        self.triangles = tri.triangles
        self.file_name = file_name

    def load_tif(self, file_name):
        " Convert a *.tif gray-level image (1-ch) to 3D image with the pixel values as Z-axis values. "
        
        vertices = []
        faces = []
        
        img = mpimg.imread(file_name)

        img_height, img_width = img.shape

        X, Y = np.meshgrid(range(img_width), range(img_height));
        tri = mtri.Triangulation(X.flatten(), Y.flatten())
        
        # fig = plt.figure(figsize=plt.figaspect(0.5))
        # ax = fig.add_subplot(1, 2, 1, projection='3d')
        # ax.plot_trisurf(X.flatten(), Y.flatten(), img.flatten(), triangles=tri.triangles, cmap=plt.cm.Spectral)

        
        width       = 350
        lower_bound = 40 - (width / 2)

        self.x = X
        self.y = Y
        self.z = img * (width / 65535) + lower_bound  #np_img * (args.width/65535) + Lower_bound Lower_bound = (args.level - (args.width/2))
        self.triangles = tri.triangles
        self.file_name = file_name
    
    def load_stl(self, file_name):
        '''Load STL CAD file
        '''
        try:
            with open(file_name, 'r') as f:
                data = f.read()

            self.load_stl_ascii(data)

        except:
            self.load_stl_binary(file_name)

    def load_stl_ascii(self, data):
        '''Load ASCII STL CAD file
        '''
        vertices = []
        faces = []
        v = []
        for i, line in enumerate(data.splitlines()):
            if i == 0 and line.strip() != 'solid':
                raise ValueError('Not valid ASCII STL file.')

            line_data = line.split()

            if line_data[0]=='facet':
                v = []

            elif line_data[0]=='vertex':
                v.append([float(line_data[1]), float(line_data[2]), float(line_data[3])])

            elif line_data[0]=='endloop':
                if len(v)==3:
                    vertices.extend(v)
                    ind = 3*len(faces)+1
                    faces.append([ind, ind+1, ind+2])

        self.data.append(Mesh(vertices, faces))

    def load_stl_binary(self, file_name):
        '''Load binary STL CAD file
        '''
        from struct import unpack
        vertices = []
        faces = []
        with open(file_name, 'rb') as f:
            header = f.read(80)
            # name = header.strip()
            n_tri = unpack('<I', f.read(4))[0]
            for i in range(n_tri):
                _normals = f.read(3*4)
                for j in range(3):
                    x = unpack('<f', f.read(4))[0]
                    y = unpack('<f', f.read(4))[0]
                    z = unpack('<f', f.read(4))[0]
                    vertices.append([x, y, z])

                j = 3*i + 1
                faces.append([j, j+1, j+2])
                _attr = f.read(2)

        self.data.append(Mesh(vertices, faces))

    def load_obj(self, file_name):
        '''Load ASCII Wavefront OBJ CAD file
        '''
        with open(file_name, 'r') as f:
            data = f.read()

        vertices = []
        faces = []
        for line in data.splitlines():
            line_data = line.split()
            if line_data:
                if line_data[0] == 'v':
                    v = [float(line_data[1]), float(line_data[2]), float(line_data[3])]
                    vertices.append(v)
                elif line_data[0] == 'f':
                    face = []
                    for i in range(1, len(line_data)):
                        s = line_data[i].replace('//','/').split('/')
                        face.append(int(s[0]))

                    faces.append(face)

        self.data.append(Mesh(vertices, faces))

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

    def plot_tif(self):

        self.clear()

        print(f"file: {self.model.file_name.split('/')[-2:]}")
        self.axes.plot_surface(self.model.x, self.model.y, self.model.z, cmap=cm.jet)
        self.axes.set(zlabel="HU", xlabel=f"{self.model.file_name.split('/')[-2]}", ylabel=f"{self.model.file_name.split('/')[-1]}")
        #self.cb = self.figure.colorbar(self.axes.plot_surface(self.model.x, self.model.y, self.model.z, cmap=cm.jet), shrink=0.5, location='left')
        self.update()

    def plot_png(self):

        self.clear()

        print(f"file: {self.model.file_name.split('/')[-2:]}")
        self.axes.plot_surface(self.model.x, self.model.y, self.model.z, cmap=cm.jet)
        self.axes.set(zlabel="HU", xlabel=f"{self.model.file_name.split('/')[-2]}", ylabel=f"{self.model.file_name.split('/')[-1]}")
        #self.cb = self.figure.colorbar(self.axes.plot_surface(self.model.x, self.model.y, self.model.z, cmap=cm.jet), shrink=0.5, location='left')
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

    def xy(self):
        self.axes.view_init(elev=90, azim=-90)
        self.update()

    def xz(self):
        self.axes.view_init(elev=0, azim=-90)
        self.update()

    def yz(self):
        self.axes.view_init(elev=0, azim=0)
        self.update()

    def reset(self):
        self.axes.view_init()
        self.update()


class Controller():

    def __init__(self, view=None):

        root = tk.Tk()
        root.title("Slice Landform Viewer")

        if view is None:
            view = View()
        view_2 = View()

        f1 = ttk.Frame(root)
        f1.pack(side=tk.TOP, anchor=tk.W)

        toolbar = [ tk.Button(f1, text="(1) Open"),
                    # tk.Button(f1, text="XY", command=view.xy),
                    # tk.Button(f1, text="XZ", command=view.xz),
                    # tk.Button(f1, text="YZ", command=view.yz),
                    tk.Button(f1, text="(1) Reset", command=view.reset),
                    tk.Button(f1, text="(2) Open"),
                    tk.Button(f1, text="(2) Reset", command=view_2.reset) ]

        # f2 = tk.Frame(f1, highlightthickness=1, highlightbackground="gray")
        # options = ["solid","wireframe","(1) solid + wireframe"]
        var = tk.StringVar()
        # o1 = ttk.OptionMenu(f2, var, options[len(options)-1], *options, command=lambda val: self.view.plot(val))
        # o1["menu"].configure(bg="white")
        # setMaxWidth(options, o1)
        # o1.pack()
        # toolbar.append(f2)

        # f3 = tk.Frame(f1, highlightthickness=1, highlightbackground="gray")
        # options2 = ["solid","wireframe","(2) solid + wireframe"]
        var2 = tk.StringVar()
        # o2 = ttk.OptionMenu(f3, var2, options2[len(options2)-1], *options2, command=lambda var2: self.view_2.plot(var2))
        # o2["menu"].configure(bg="white")
        # setMaxWidth(options2, o2)
        # o2.pack()
        # toolbar.append(f3)

        toolbar[0].config(command=lambda: self.open(var))
        toolbar[2].config(command=lambda: self.open2(var2))

        [obj.pack(side=tk.LEFT, anchor=tk.W) for obj in toolbar]

        canvas = FigureCanvasTkAgg(view.figure, root)
        canvas.mpl_connect('button_press_event', view.axes._button_press)
        canvas.mpl_connect('button_release_event', view.axes._button_release)
        canvas.mpl_connect('motion_notify_event', view.axes._on_move)
        canvas.draw()
        #canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        #mpl_toolbar = NavigationToolbar2Tk(canvas, root)
        #mpl_toolbar.update()
        canvas._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        canvas2 = FigureCanvasTkAgg(view_2.figure, root)
        canvas2.mpl_connect('button_press_event', view_2.axes._button_press)
        canvas2.mpl_connect('button_release_event', view_2.axes._button_release)
        canvas2.mpl_connect('motion_notify_event', view_2.axes._on_move)
        canvas2.draw()
        #canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)        
        #mpl_toolbar2 = NavigationToolbar2Tk(canvas2, root)
        #mpl_toolbar2.update()
        canvas2._tkcanvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # menubar = tk.Menu( root )
        # file_menu = tk.Menu(menubar, tearoff=0)
        # file_menu.add_command(label="Open...", command=lambda: self.open(var))
        # file_menu.add_command(label="Open (2)...", command=lambda: self.open2(var2))
        # file_menu.add_command(label="Exit", command=self.exit)
        # menubar.add_cascade(label="File", menu=file_menu)
        # root.config(menu=menubar)

        self.root = root
        view.canvas = canvas
        #view.toolbar = mpl_toolbar
        self.view = view
        self.model = view.model

        view_2.canvas = canvas2
        #view_2.toolbar = mpl_toolbar2
        self.view_2 = view_2
        self.model2 = view_2.model

    def render(self):
        self.root.mainloop()

    def open(self, var):
        file_name = askopenfilename( title = "Select file to open",
                                     filetypes = (  ("all files","*.*"),
                                                    ("CAD files","*.obj;*.stl") ))
        self.model.clear()
        self.model.load_file(file_name)

        if file_name.lower().endswith('.tif'):

            self.view.plot_tif()
        
        elif file_name.lower().endswith('.png'):

            self.view.plot_png()

        else:
            self.view.plot(var.get())   
    
    def open2(self, var2):
        file_name = askopenfilename( title = "Select file to open",
                                     filetypes = (  ("all files","*.*"),
                                                    ("CAD files","*.obj;*.stl") ))
        self.model2.clear()
        self.model2.load_file(file_name)

        if file_name.lower().endswith('.tif'):

            self.view_2.plot_tif()

        elif file_name.lower().endswith('.png'):

            self.view_2.plot_png()
        
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

    def __init__(self, model=None, view=None, controller=None):
        file_name = None
        if len(sys.argv) >= 2:
            file_name = sys.argv[1]

        if model is None:
            model = Model(file_name)

        if view is None:
            view = View(model)

        if controller is None:
            controller = Controller(view)

        self.model = model
        self.view = view
        self.controller = controller

    def start(self):
        self.controller.render()


if __name__ == "__main__":

    app = App()
    app.start()
