import argparse
import json
import os
import glob

import sys
import numpy as np
import matplotlib.pyplot as plt

API_description = """
***** Radiomics Analysis Platform  *****
API Name: Clickable Scatter Plot
Version:    1.0
Developer: Alvin Li
Email:     d05548014@ntu.edu.tw
****************************************

"""

_pwd_ = os.getcwd()

parser = argparse.ArgumentParser(prog='test_clickable_scatterplot.py',
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=API_description)

parser.add_argument('-Table', 
                    action = 'store', 
                    type = str, 
                    help = 'The absolute path to the DATA TABLE (*.json).')

parser.add_argument('-Feature', 
                    action = 'store', 
                    type = str, 
                    help = 'The absolute path to the Feature TABLE (*.txt).')

parser.add_argument('-Root', 
                    action = 'store', 
                    type = str, 
                    help = 'The absolute path to the image root.')

args = parser.parse_args()

colors = ['navy', 'darkorange']

print(f"Loading {args.Table}")

json_file = open( args.Table, 'r' )
JTable = json.load(json_file)
json_file.close()

Label_0 = JTable["0"]
Label_1 = JTable["1"]

X0 = Label_0["X"]
Y0 = Label_0["Y"]
Tumor0 = Label_0["Tumor"]

X1 = Label_1["X"]
Y1 = Label_1["Y"]
Tumor1 = Label_1["Tumor"]

ScatterData = []
ScatterColor = []
ScatterTumor = []

fig, ax = plt.subplots()

N = len(X0)
for n in range(N):
    x = X0[n]
    y = Y0[n]
    coord = [x, y]
    ScatterData.append(coord)
    ScatterColor.append(colors[0])
    ScatterTumor.append(Tumor0[n])

ax.scatter( X0, 
            Y0, 
            color = colors[0], 
            label = "Label 0",
            picker = 5, 
            lw=2)

N = len(X1)
for n in range(N):
    x = X1[n]
    y = Y1[n]
    coord = [x, y]
    ScatterData.append(coord)
    ScatterColor.append(colors[1])
    ScatterTumor.append(Tumor1[n])

ScatterData = np.array(ScatterData)
ScatterColor = np.array(ScatterColor)
ScatterTumor = np.array(ScatterTumor)

ax.scatter( X1, 
            Y1, 
            color = colors[1], 
            label = "Label 1",
            picker = 5, 
            lw=2)

# ax.scatter( ScatterData[:,0], 
#             ScatterData[:,1], 
#             color = ScatterColor, 
#             #label = ScatterColor,
#             picker = 5, 
#             lw=2)

# testData = np.array([[0,0], [0.1, 0], [0, 0.3], [-0.4, 0], [0, -0.5]])
# fig, ax = plt.subplots()
# coll = ax.scatter(testData[:,0], testData[:,1], color=["blue"]*len(testData), picker = 5, s=[50]*len(testData))
plt.grid(True)
plt.title(os.path.splitext(os.path.basename(args.Table))[0])
plt.gca().set(aspect='equal')
plt.legend()
# plt.axis([-0.5, 0.5, -0.5, 0.5])

PY_tumor_display = os.path.join(_pwd_, "tumor_display.py")

json_file = open(args.Feature, 'r')
FTable = json.load(json_file)
json_file.close()

def on_pick(event):
    print(ScatterTumor[event.ind][0], f":{ScatterData[event.ind]}", "clicked")
    if (os.path.exists(PY_tumor_display)):
        patient, slice = ScatterTumor[event.ind][0].split(":")
        folder = glob.glob(os.path.join(args.Root, f'{patient}_*'))[0]
        o_image  = os.path.join(folder, f'{slice}.tif')
        o_mask   = os.path.join(folder, f'{slice}.png')
        os.system(f"python3 {PY_tumor_display} -R {o_mask} -L {o_image} &")
        os.system(f"python3 {PY_tumor_display} -R {FTable[patient]['Radiomics'][slice]['mask']} -L {FTable[patient]['Radiomics'][slice]['image']} &")
    # coll._facecolors[event.ind,:] = (1, 0, 0, 1)
    # coll._edgecolors[event.ind,:] = (1, 0, 0, 1)
    # fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()
