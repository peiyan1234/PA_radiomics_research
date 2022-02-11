import os 
import nrrd 

from PIL import Image
import numpy as np

CT_VOL = "/media/share/DATA/Radiomics/Adrenal/data/CTimages/PA135_20180929_A.nrrd"
CT_SEG = "/media/share/DATA/Radiomics/Adrenal/data/Annotations/PA135_20180929_A.seg.nrrd"

# abdomen
# soft tissues W:400 L:50
CT_WindowWidth = 400.0
CT_WidnowLevel = 50.0

HU_upperbound = CT_WidnowLevel + ( CT_WindowWidth / 2.0 )
HU_lowerbound = CT_WidnowLevel - ( CT_WindowWidth / 2.0 )

print(f"Crop ROI from {CT_VOL}")
print(f"Crop ROI refer to {CT_SEG}")

vol_data, vol_meta = nrrd.read(CT_VOL)
seg_data, seg_meta = nrrd.read(CT_SEG)

print(vol_meta)
print(vol_data.shape)
print(seg_meta)
print(seg_data.shape)

seg_dimension = seg_meta['dimension']
if ( 3==seg_dimension ):
    layer_n = 1
    x_size, y_size, z_size = seg_meta['sizes']
else:
    layer_n, x_size, y_size, z_size = seg_meta['sizes']

print(f"Segmentation Matrix Information: {seg_meta['sizes']}")
print(f"Segmentation Layer count: {layer_n}")
print(f"Segmentation 3D volume: {x_size, y_size, z_size}")

print("Deal with adrenal")
Seg_info = [ layer_n, x_size, y_size, z_size, seg_data, seg_meta, vol_data ]
vol = vol_data
mask = seg_data

x_max, x_min = -1, x_size
y_max, y_min = -1, y_size
z_max, z_min = -1, z_size

seg_labels = []

for i in range(100):
    c = f"Segment{i}_Name"
    if c in seg_meta.keys():
        seg_extent = seg_meta[f"Segment{i}_Extent"]
        print(seg_extent)
        seg_labels.append( (seg_meta[f"Segment{i}_Layer"], seg_meta[f"Segment{i}_LabelValue"]) )
        xyz_boundary = seg_extent.split()
        x_min = int(xyz_boundary[0]) if ( int(xyz_boundary[0]) < x_min ) else x_min
        x_max = int(xyz_boundary[1]) if ( int(xyz_boundary[1]) > x_max ) else x_max
        y_min = int(xyz_boundary[2]) if ( int(xyz_boundary[2]) < y_min ) else y_min
        y_max = int(xyz_boundary[3]) if ( int(xyz_boundary[3]) > y_max ) else y_max
        z_min = int(xyz_boundary[4]) if ( int(xyz_boundary[4]) < z_min ) else z_min
        z_max = int(xyz_boundary[5]) if ( int(xyz_boundary[5]) > z_max ) else z_max

vol_LB = vol < HU_lowerbound
vol_HB = vol > HU_upperbound

vol[vol_LB] = HU_lowerbound
vol[vol_HB] = HU_upperbound

vol = 255 * ( vol - HU_lowerbound ) / CT_WindowWidth

vol_block = vol[ x_min : x_max+1,
                 y_min : y_max+1, 
                 z_min : z_max+1 ]

print(x_min, x_max, x_max+1-x_min)
print(y_min, y_max, y_max+1-y_min)
print(z_min, z_max, z_max+1-z_min)
print(vol.shape)
#print(vol_block.shape)
print(vol[0,0,z_max])
