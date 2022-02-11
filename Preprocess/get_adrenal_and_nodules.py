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

import nrrd
from PIL import Image
import numpy as np

API_Description = """
***** Radiomics Analysis Platform  *****
API Name: Get Adrenal and Nodules from CT and Annotations
Version:    1.0
Developer: Pei-Yan Li
Email:     d05548014@ntu.edu.tw
****************************************
"""

parser = argparse.ArgumentParser(prog = 'get_adrenal_and_nodules.py',
                                 formatter_class = argparse.RawDescriptionHelpFormatter,
                                 description = API_Description)

parser.add_argument('-D', action = 'store', type = str, help='the path to the datasheet.csv')
parser.add_argument('-S', action = 'store', type = str, help='the path to the folders of CTimages and Annotations')
args = parser.parse_args()

def read_csv():

    if ( os.path.isfile( args.D ) ):
        with open(args.D) as csv_file:
            No_line = 1
            table = []
            for row in csv.reader(csv_file, delimiter=','):
                if ( No_line > 2 ):
                    patient_ID = row[0]
                    N_phase = True if (row[2]=='V' or row[2]=='v') else False
                    A_phase = True if (row[3]=='V' or row[3]=='v') else False
                    V_phase = True if (row[4]=='V' or row[4]=='v') else False
                    table.append( (patient_ID, N_phase, A_phase, V_phase ) )
                No_line +=1

            if ( len(table) > 0 ):
                return table
    return False

def get_left_adrenal(*Seg_info, outputfolder):

    layer_n, x_size, y_size, z_size, mask, seg_meta, vol = Seg_info

    x_max, x_min = -1, x_size
    y_max, y_min = -1, y_size
    z_max, z_min = -1, z_size

    seg_labels = []

    for i in range(100):
        c = f"Segment{i}_Name"
        if ( c in seg_meta.keys() ):
            pass
        else:
            continue
        if ( "Left" in seg_meta[c] or "left" in seg_meta[c] ):
            pass
        else:
            continue
            
        seg_extent = seg_meta[f"Segment{i}_Extent"]
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

    if ( 1 < layer_n ):
        seg = np.zeros( (x_size, y_size, z_size) )
        for ele in seg_labels:
            seg_layer, seg_label = ele
            tmsk = mask[ int(seg_layer), :, :, : ]
            buf = np.zeros(tmsk.shape, order='F')
            buf[ tmsk==int(seg_label) ] = 1
            seg += buf
    else:
        seg = np.zeros( (x_size, y_size, z_size) )
        for ele in seg_labels:
            seg_layer, seg_label = ele
            buf = np.zeros(mask.shape, order='F')
            buf[ mask==int(seg_label) ] = 1
            seg += buf

    seg[ seg >= 1 ] = 255

    seg_block = seg[ x_min : x_max+1,
                     y_min : y_max+1,
                     z_min : z_max+1 ]
    
    save_folder = os.path.join( outputfolder, "LeftAdrenal" )
    try:
        os.mkdir( save_folder )
    except:
        pass

    for z in range(z_max-z_min+1):
        v_slice = vol_block[:, :, z]
        m_slice = seg_block[:, :, z]

        v_slice = v_slice.transpose()
        m_slice = m_slice.transpose()

        msk = m_slice != 255
        v_slice[ msk ] = 0
        if ( 0==np.count_nonzero(v_slice) ):
            continue

        seg_path = os.path.join( save_folder, f"{z}".zfill(3) + ".png" )
        Image.fromarray( v_slice.astype(np.uint8) ).save( seg_path )
        vol_block[:, :, z] = v_slice.transpose() 
    
    npy_path = os.path.join( save_folder, f"fullimages.npy" )
    np.save(npy_path, vol_block.astype(np.uint8))

    return True

def get_right_adrenal(*Seg_info, outputfolder):

    layer_n, x_size, y_size, z_size, mask, seg_meta, vol = Seg_info

    x_max, x_min = -1, x_size
    y_max, y_min = -1, y_size
    z_max, z_min = -1, z_size

    seg_labels = []

    for i in range(100):
        c = f"Segment{i}_Name"
        if ( c in seg_meta.keys() ):
            pass
        else:
            continue
        if ( "Right" in seg_meta[c] or "right" in seg_meta[c] ):
            pass
        else:
            continue
            
        seg_extent = seg_meta[f"Segment{i}_Extent"]
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

    if ( 1 < layer_n ):
        seg = np.zeros( (x_size, y_size, z_size) )
        for ele in seg_labels:
            seg_layer, seg_label = ele
            tmsk = mask[ int(seg_layer), :, :, : ]
            buf = np.zeros(tmsk.shape, order='F')
            buf[ tmsk==int(seg_label) ] = 1
            seg += buf
    else:
        seg = np.zeros( (x_size, y_size, z_size) )
        for ele in seg_labels:
            seg_layer, seg_label = ele
            buf = np.zeros(mask.shape, order='F')
            buf[ mask==int(seg_label) ] = 1
            seg += buf

    seg[ seg >= 1 ] = 255

    seg_block = seg[ x_min : x_max+1,
                     y_min : y_max+1,
                     z_min : z_max+1 ]
    
    save_folder = os.path.join( outputfolder, "RightAdrenal" )
    try:
        os.mkdir( save_folder )
    except:
        pass

    for z in range(z_max-z_min+1):
        v_slice = vol_block[:, :, z]
        m_slice = seg_block[:, :, z]

        v_slice = v_slice.transpose()
        m_slice = m_slice.transpose()

        msk = m_slice != 255
        v_slice[ msk ] = 0
        if ( 0==np.count_nonzero(v_slice) ):
            continue

        seg_path = os.path.join( save_folder, f"{z}".zfill(3) + ".png" )
        Image.fromarray( v_slice.astype(np.uint8) ).save( seg_path )
        vol_block[:, :, z] = v_slice.transpose() 
    
    npy_path = os.path.join( save_folder, f"fullimages.npy" )
    np.save(npy_path, vol_block.astype(np.uint8))

    return True

def get_adrenal(*Seg_info, outputfolder):

    layer_n, x_size, y_size, z_size, mask, seg_meta, vol = Seg_info

    x_max, x_min = -1, x_size
    y_max, y_min = -1, y_size
    z_max, z_min = -1, z_size

    seg_labels = []

    for i in range(100):
        c = f"Segment{i}_Name"
        if c in seg_meta.keys():
            seg_extent = seg_meta[f"Segment{i}_Extent"]
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

    if ( 1 < layer_n ):
        seg = np.zeros( (x_size, y_size, z_size) )
        for ele in seg_labels:
            seg_layer, seg_label = ele
            tmsk = mask[ int(seg_layer), :, :, : ]
            buf = np.zeros(tmsk.shape, order='F')
            buf[ tmsk==int(seg_label) ] = 1
            seg += buf
    else:
        seg = np.zeros( (x_size, y_size, z_size) )
        for ele in seg_labels:
            seg_layer, seg_label = ele
            buf = np.zeros(mask.shape, order='F')
            buf[ mask==int(seg_label) ] = 1
            seg += buf
    
    seg[ seg >= 1 ] = 255

    seg_block = seg[ x_min : x_max+1,
                     y_min : y_max+1,
                     z_min : z_max+1 ]
    
    save_folder = os.path.join( outputfolder, "Adrenal" )
    try:
        os.mkdir( save_folder )
    except:
        pass

    for z in range(z_max-z_min+1):
        v_slice = vol_block[:, :, z]
        m_slice = seg_block[:, :, z]

        v_slice = v_slice.transpose()
        m_slice = m_slice.transpose()

        msk = m_slice != 255
        v_slice[ msk ] = 0
        if ( 0==np.count_nonzero(v_slice) ):
            continue

        seg_path = os.path.join( save_folder, f"{z}".zfill(3) + ".png" )
        Image.fromarray( v_slice.astype(np.uint8) ).save( seg_path )
        vol_block[:, :, z] = v_slice.transpose()
    
    npy_path = os.path.join( save_folder, f"fullimages.npy" )
    np.save(npy_path, vol_block.astype(np.uint8))

    return True

def get_nodules(*Seg_info, vol, mask, outputfolder):
    
    layer_n, seg_name, seg_layer, seg_label, seg_extent = Seg_info
    
    xyz_boundary = seg_extent.split()
    x_min = int(xyz_boundary[0])
    x_max = int(xyz_boundary[1])
    y_min = int(xyz_boundary[2])
    y_max = int(xyz_boundary[3])
    z_min = int(xyz_boundary[4])
    z_max = int(xyz_boundary[5])

    vol_LB = vol < HU_lowerbound
    vol_HB = vol > HU_upperbound

    vol[vol_LB] = HU_lowerbound
    vol[vol_HB] = HU_upperbound

    vol = 255 * ( vol - HU_lowerbound ) / CT_WindowWidth

    vol_block = vol[ x_min : x_max+1,
                     y_min : y_max+1,
                     z_min : z_max+1 ]

    if ( 1 < layer_n ):
        seg = mask[ int(seg_layer), :, :, : ]
    elif ( 1 == layer_n ):
        seg = mask
    else:
        return False
    
    seg_block = seg[ x_min : x_max+1,
                     y_min : y_max+1,
                     z_min : z_max+1 ]
    
    save_folder = os.path.join( outputfolder, seg_name )
    try:
        os.mkdir( save_folder )
    except:
        pass

    for z in range(z_max-z_min+1):
        v_slice = vol_block[:, :, z]
        m_slice = seg_block[:, :, z]

        v_slice = v_slice.transpose()
        m_slice = m_slice.transpose()

        msk = m_slice != int(seg_label)
        v_slice[ msk ] = 0
        if ( 0==np.count_nonzero(v_slice) ):
            continue

        seg_path = os.path.join( save_folder, f"{z}".zfill(3) + ".png" )
        Image.fromarray( v_slice.astype(np.uint8) ).save( seg_path )
        vol_block[:, :, z] = v_slice.transpose() 
    
    npy_path = os.path.join( save_folder, f"fullimages.npy" )
    np.save(npy_path, vol_block.astype(np.uint8))

    return True

def crop_annotations(outputfolder, image, annotation):

    print(f"Crop labels from {image}")
    print(f"Save in {outputfolder}")

    vol_data, vol_meta = nrrd.read(image)
    seg_data, seg_meta = nrrd.read(annotation)

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
    if ( get_adrenal(*Seg_info, outputfolder=outputfolder) ):
        print("OK!")
    else:
        print("Failed!")
        print("Quit!")
        quit()

    print("Deal with right adrenal")
    if ( get_right_adrenal(*Seg_info, outputfolder=outputfolder) ):
        print("OK!")
    else:
        print("Failed!")
        print("Quit!")
        quit()

    print("Deal with left adrenal")
    if ( get_left_adrenal(*Seg_info, outputfolder=outputfolder) ):
        print("OK!")
    else:
        print("Failed!")
        print("Quit!")
        quit()
    
    for i in range(100):
        c = f"Segment{i}_Name"
        if c in seg_meta.keys():
            if "adrenal" in seg_meta[c] or "Adrenal" in seg_meta[c]:
                pass
            else:
                print(f"Deal with {seg_meta[c]}")
                Seg_Layer = seg_meta[f"Segment{i}_Layer"]
                Seg_LabelValue = seg_meta[f"Segment{i}_LabelValue"]
                Seg_Extent = seg_meta[f"Segment{i}_Extent"]
                Seg_info = [layer_n, seg_meta[c], Seg_Layer, Seg_LabelValue, Seg_Extent]
                if ( get_nodules(*Seg_info, vol=vol_data, 
                                        mask=seg_data, 
                                        outputfolder=outputfolder) ):
                    print(f"OK!")
                else:
                    print(f"Failed! {seg_meta[c]}")
                    print("Quit!")
                    quit()

if __name__ == '__main__':

    if ( os.path.isdir(args.S) ):
        try:
            os.mkdir( os.path.join(args.S, 'output_adrenal_and_nodules') )
        except:
            pass
    else:
        print( f" args.S is invalid: {args.S}" )
        quit()

    csv_table = read_csv()
    if ( False != csv_table ):
        CTimagefolder = os.path.join( args.S, 'CTimages' )
        CTlabelsfolder= os.path.join( args.S, 'Annotations' )

        outputfolder = os.path.join(args.S, 'output_adrenal_and_nodules')
        
        for patient_ID, N_phase, A_phase, V_phase in csv_table:

            patientfolder = os.path.join( outputfolder, patient_ID )

            if ( N_phase or A_phase or V_phase ):
                pass
            else:
                continue

            try:
                os.mkdir(patientfolder)
            except:
                pass
            print(f"{patient_ID}, N: {N_phase}, A: {A_phase}, V: {V_phase}")
            
            #if N_phase:
            N_images = glob.glob( os.path.join(CTimagefolder, f"{patient_ID}_*N.nrrd") )
            N_labels = glob.glob( os.path.join(CTlabelsfolder, f"{patient_ID}_*N.seg.nrrd") )
            N_folder = os.path.join(patientfolder, 'N_phase')
            if ( N_phase and len(N_images) > 0 and len(N_labels) > 0 ):
                try:
                    os.mkdir( N_folder )
                except:
                    pass
                crop_annotations( N_folder, N_images[0], N_labels[0] )

            #if A_phase:
            A_images = glob.glob( os.path.join(CTimagefolder, f"{patient_ID}_*A.nrrd") )
            A_labels = glob.glob( os.path.join(CTlabelsfolder, f"{patient_ID}_*A.seg.nrrd") )
            A_folder = os.path.join(patientfolder, 'A_phase')
            if ( A_phase and len(A_images) > 0 and len(A_labels) > 0 ):
                try:
                    os.mkdir( A_folder )
                except:
                    pass
                crop_annotations( A_folder, A_images[0], A_labels[0] )

            #if V_phase:
            V_images = glob.glob( os.path.join(CTimagefolder, f"{patient_ID}_*V.nrrd") )
            V_labels = glob.glob( os.path.join(CTlabelsfolder, f"{patient_ID}_*V.seg.nrrd") )
            V_folder = os.path.join(patientfolder, 'V_phase')
            if ( V_phase and len(V_images) > 0 and len(V_labels) > 0 ):
                try:
                    os.mkdir( V_folder )
                except:
                    pass
                crop_annotations( V_folder, V_images[0], V_labels[0] )



    
