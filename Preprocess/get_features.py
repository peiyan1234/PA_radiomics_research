import argparse
import csv
import json
import os
import glob

# abdomen
# soft tissues W:400 L:50
CT_WindowWidth = 400.0
CT_WidnowLevel = 50.0

HU_upperbound = CT_WidnowLevel + ( CT_WindowWidth / 2.0 )
HU_lowerbound = CT_WidnowLevel - ( CT_WindowWidth / 2.0 )

# mm
x_pixel_length = 0.405 
y_pixel_length = 0.405
z_pixel_length = 1.0
UNIT_VOLUME = z_pixel_length * y_pixel_length * x_pixel_length

hyperparameters = {}
hyperparameters['setting'] = {}
hyperparameters['setting']['voxelArrayShift'] = 0 #HU_lowerbound

TYPE = ""
FEATURES = {}
FEATURES_LIST = []

import nrrd
from PIL import Image
import numpy as np

import radiomics
from radiomics import featureextractor
import SimpleITK as sitk

API_Description = """
***** Radiomics Analysis Platform  *****
API Name: Get Features from CT and Annotations
Version:    1.0
Developer: Pei-Yan Li
Email:     d05548014@ntu.edu.tw
****************************************
"""

_pwd_ = os.getcwd()

parser = argparse.ArgumentParser(prog = 'get_features.py',
                                 formatter_class = argparse.RawDescriptionHelpFormatter,
                                 description = API_Description)

parser.add_argument('-D', action = 'store', type = str, help='the path to the datasheet.csv')
parser.add_argument('-S', action = 'store', type = str, help='the path to the folders of CTimages and Annotations')
parser.add_argument('-csv', action = 'store', type = bool, help='convert feature.json to feature.csv')
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
                    type    = row[14]
                    table.append( (patient_ID, N_phase, A_phase, V_phase, type ) )
                No_line +=1

            if ( len(table) > 0 ):
                return table
    return False

def save_csv():

    csv_path = os.path.join(FEATURES_TABLE_OUTPUT, 'Features.csv')
    csv_file = open(csv_path, 'w')    
    writer = csv.writer(csv_file, dialect='excel')
    headers = []
    headers.append('Subject')
    headers.append('Lateralization')
    headers.append('Phase')
    headers.append('Location')
    headers.append('Filter')

    #FEATURES[ID]['Phase'][phase]["RightAdrenal"] = { 'Original' : F['Original'] }
    for feature_type in FEATURES_LIST:
        headers.append(feature_type)
    writer.writerow(headers)

    for id in sorted(FEATURES.keys()):
        for ph in sorted(list(FEATURES[id]['Phase'].keys())):
            for loc in sorted(list(FEATURES[id]['Phase'][ph].keys())):
                for fil in sorted(list(FEATURES[id]['Phase'][ph][loc].keys())):
                    _line_ = []
                    _line_.append(id)
                    _line_.append(FEATURES[id]["lateralization"])
                    _line_.append(ph)
                    _line_.append(loc)
                    _line_.append(fil)
                    for feature_type in FEATURES_LIST:
                        _line_.append(FEATURES[id]['Phase'][ph][loc][fil][feature_type])
                writer.writerow(_line_)
    
    csv_file.close()
    
    a = zip(*csv.reader(open(csv_path, "r")))    
    csv.writer(open(csv_path, "w")).writerows(a)

def get_left_adrenal_features(*Seg_info, outputfolder, phase, ID):

    global FEATURES

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
    
    '''
    {'firstorder': <class 'radiomics.firstorder.RadiomicsFirstOrder'>, 
     'glcm': <class 'radiomics.glcm.RadiomicsGLCM'>, 
     'gldm': <class 'radiomics.gldm.RadiomicsGLDM'>, 
     'glrlm': <class 'radiomics.glrlm.RadiomicsGLRLM'>, 
     'glszm': <class 'radiomics.glszm.RadiomicsGLSZM'>, 
     'ngtdm': <class 'radiomics.ngtdm.RadiomicsNGTDM'>, 
     'shape': <class 'radiomics.shape.RadiomicsShape'>, 
     'shape2D': <class 'radiomics.shape2D.RadiomicsShape2D'>}
    '''

    extractor = featureextractor.RadiomicsFeatureExtractor(**hyperparameters)
    #extractor.enableImageTypeByName('LBP3D')
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')

    IMG = sitk.GetImageFromArray(vol_block)
    MSK = sitk.GetImageFromArray(seg_block)

    features = extractor.execute(IMG, MSK, 255)

    F = {}
    F['Original'] = {}
    F['LBP3D'] = {}

    for key in features.keys():
        if 'diagnostics' in key:
            continue
        if 'original' in key:
            F['Original'][key.split('original_')[1]] = float(features[key])
            continue
        if 'lbp-3D' in key:
            F['LBP3D'][key.split('lbp-3D-')[1]] = float(features[key])
            continue

    volume = UNIT_VOLUME * np.count_nonzero( seg_block )

    Feature_Table = {}
    Feature_Table['lateralization'] = TYPE
    Feature_Table['Radiomics'] = F
    Feature_Table['volume'] = volume

    FEATURES[ID]['Phase'][phase]["LeftAdrenal"] = { 'Original' : F['Original'] }

    json_path = os.path.join( save_folder, 'Features.json' )
    json_file = open(json_path, 'w')
    json_content = json.dumps(Feature_Table, indent = 4)
    json_file.writelines(json_content)
    json_file.close()
    
    return True

def get_right_adrenal_features(*Seg_info, outputfolder, phase, ID):

    global FEATURES

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
    
    '''
    {'firstorder': <class 'radiomics.firstorder.RadiomicsFirstOrder'>, 
     'glcm': <class 'radiomics.glcm.RadiomicsGLCM'>, 
     'gldm': <class 'radiomics.gldm.RadiomicsGLDM'>, 
     'glrlm': <class 'radiomics.glrlm.RadiomicsGLRLM'>, 
     'glszm': <class 'radiomics.glszm.RadiomicsGLSZM'>, 
     'ngtdm': <class 'radiomics.ngtdm.RadiomicsNGTDM'>, 
     'shape': <class 'radiomics.shape.RadiomicsShape'>, 
     'shape2D': <class 'radiomics.shape2D.RadiomicsShape2D'>}
    '''

    extractor = featureextractor.RadiomicsFeatureExtractor(**hyperparameters)
    #extractor.enableImageTypeByName('LBP3D')
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')

    IMG = sitk.GetImageFromArray(vol_block)
    MSK = sitk.GetImageFromArray(seg_block)

    features = extractor.execute(IMG, MSK, 255)

    F = {}
    F['Original'] = {}
    F['LBP3D'] = {}

    for key in features.keys():
        if 'diagnostics' in key:
            continue
        if 'original' in key:
            F['Original'][key.split('original_')[1]] = float(features[key])
            continue
        if 'lbp-3D' in key:
            F['LBP3D'][key.split('lbp-3D-')[1]] = float(features[key])
            continue

    volume = UNIT_VOLUME * np.count_nonzero( seg_block )

    Feature_Table = {}
    Feature_Table['lateralization'] = TYPE
    Feature_Table['Radiomics'] = F
    Feature_Table['volume'] = volume

    FEATURES[ID]['Phase'][phase]["RightAdrenal"] = { 'Original' : F['Original'] }

    json_path = os.path.join( save_folder, 'Features.json' )
    json_file = open(json_path, 'w')
    json_content = json.dumps(Feature_Table, indent = 4)
    json_file.writelines(json_content)
    json_file.close()

    return True

def get_adrenal_features(*Seg_info, outputfolder, phase, ID):

    global FEATURES

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

    '''
    {'firstorder': <class 'radiomics.firstorder.RadiomicsFirstOrder'>, 
     'glcm': <class 'radiomics.glcm.RadiomicsGLCM'>, 
     'gldm': <class 'radiomics.gldm.RadiomicsGLDM'>, 
     'glrlm': <class 'radiomics.glrlm.RadiomicsGLRLM'>, 
     'glszm': <class 'radiomics.glszm.RadiomicsGLSZM'>, 
     'ngtdm': <class 'radiomics.ngtdm.RadiomicsNGTDM'>, 
     'shape': <class 'radiomics.shape.RadiomicsShape'>, 
     'shape2D': <class 'radiomics.shape2D.RadiomicsShape2D'>}
    '''

    extractor = featureextractor.RadiomicsFeatureExtractor(**hyperparameters)
    #extractor.enableImageTypeByName('LBP3D')
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')

    IMG = sitk.GetImageFromArray(vol_block)
    MSK = sitk.GetImageFromArray(seg_block)

    features = extractor.execute(IMG, MSK, 255)

    F = {}
    F['Original'] = {}
    F['LBP3D'] = {}

    for key in features.keys():
        if 'diagnostics' in key:
            continue
        if 'original' in key:
            F['Original'][key.split('original_')[1]] = float(features[key])
            continue
        if 'lbp-3D' in key:
            F['LBP3D'][key.split('lbp-3D-')[1]] = float(features[key])
            continue

    volume = UNIT_VOLUME * np.count_nonzero( seg_block )

    Feature_Table = {}
    Feature_Table['lateralization'] = TYPE
    Feature_Table['Radiomics'] = F
    Feature_Table['volume'] = volume

    FEATURES[ID]['Phase'][phase]["Adrenal"] = { 'Original' : F['Original'] }

    global FEATURES_LIST
    FEATURES_LIST = list(F['Original'].keys())

    json_path = os.path.join( save_folder, 'Features.json' )
    json_file = open(json_path, 'w')
    json_content = json.dumps(Feature_Table, indent = 4)
    json_file.writelines(json_content)
    json_file.close()

    return True

def get_nodules_features(*Seg_info, vol, mask, outputfolder, phase, ID):

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

    seg[ seg == seg_label ] = 255
    
    seg_block = seg[ x_min : x_max+1,
                     y_min : y_max+1,
                     z_min : z_max+1 ]
    
    save_folder = os.path.join( outputfolder, seg_name )
    try:
        os.mkdir( save_folder )
    except:
        json_path = os.path.join( save_folder, 'Features.json' )
        if ( os.path.isfile(json_path) and json.load(open(json_path, 'r')) ):
            return True

    '''
    {'firstorder': <class 'radiomics.firstorder.RadiomicsFirstOrder'>, 
     'glcm': <class 'radiomics.glcm.RadiomicsGLCM'>, 
     'gldm': <class 'radiomics.gldm.RadiomicsGLDM'>, 
     'glrlm': <class 'radiomics.glrlm.RadiomicsGLRLM'>, 
     'glszm': <class 'radiomics.glszm.RadiomicsGLSZM'>, 
     'ngtdm': <class 'radiomics.ngtdm.RadiomicsNGTDM'>, 
     'shape': <class 'radiomics.shape.RadiomicsShape'>, 
     'shape2D': <class 'radiomics.shape2D.RadiomicsShape2D'>}
    '''

    extractor = featureextractor.RadiomicsFeatureExtractor(**hyperparameters)
    #extractor.enableImageTypeByName('LBP3D')
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('shape')

    IMG = sitk.GetImageFromArray(vol_block)
    MSK = sitk.GetImageFromArray(seg_block)

    try:
        features = extractor.execute(IMG, MSK, int(seg_label))
    except:
        return True

    F = {}
    F['Original'] = {}
    F['LBP3D'] = {}

    for key in features.keys():
        if 'diagnostics' in key:
            continue
        if 'original' in key:
            F['Original'][key.split('original_')[1]] = float(features[key])
            continue
        if 'lbp-3D' in key:
            F['LBP3D'][key.split('lbp-3D-')[1]] = float(features[key])
            continue

    volume = UNIT_VOLUME * np.count_nonzero( seg_block )

    Feature_Table = {}
    Feature_Table['lateralization'] = TYPE
    Feature_Table['Radiomics'] = F
    Feature_Table['volume'] = volume

    json_path = os.path.join( save_folder, 'Features.json' )
    json_file = open(json_path, 'w')
    json_content = json.dumps(Feature_Table, indent = 4)
    json_file.writelines(json_content)
    json_file.close()

    return True

def crop_annotations(phase, ID, outputfolder, image, annotation):

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
    if ( get_adrenal_features(*Seg_info, outputfolder=outputfolder, phase=phase, ID=ID ) ):
        print("OK!")
    else:
        print("Failed!")
        print("Quit!")
        quit()

    print("Deal with right adrenal")
    if ( get_right_adrenal_features(*Seg_info, outputfolder=outputfolder, phase=phase, ID=ID ) ):
        print("OK!")
    else:
        print("Failed!")
        print("Quit!")
        quit()

    print("Deal with left adrenal")
    if ( get_left_adrenal_features(*Seg_info, outputfolder=outputfolder, phase=phase, ID=ID ) ):
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
                if ( get_nodules_features(*Seg_info, vol=vol_data, 
                                          mask=seg_data, 
                                          outputfolder=outputfolder,
                                          phase=phase, ID=ID ) ):
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
        try:
            os.mkdir(  os.path.join(args.S, 'output_features') )
        except:
            pass

    else:
        print( f" args.S is invalid: {args.S}" )
        quit()

    if ( args.csv == True ):
        FEATURES_TABLE_OUTPUT = os.path.join(args.S, 'output_features')
        json_path = os.path.join( FEATURES_TABLE_OUTPUT, 'Features.json' )
        json_file = open(json_path, 'r')
        FEATURES = json.load(json_file)
        FEATURES_LIST = sorted(list(FEATURES["PA1"]["Phase"]["N_phase"]["Adrenal"]["Original"].keys()))
        save_csv()
        quit()

    csv_table = read_csv()
    if ( False != csv_table ):
        CTimagefolder = os.path.join( args.S, 'CTimages' )
        CTlabelsfolder= os.path.join( args.S, 'Annotations' )

        outputfolder = os.path.join(args.S, 'output_adrenal_and_nodules')
        FEATURES_TABLE_OUTPUT = os.path.join(args.S, 'output_features')
        
        for patient_ID, N_phase, A_phase, V_phase, type in csv_table:
            
            patientfolder = os.path.join( outputfolder, patient_ID )
            TYPE = type
            FEATURES[patient_ID] = {}
            FEATURES[patient_ID]['lateralization'] = type
            FEATURES[patient_ID]['Phase'] = {}
            FEATURES[patient_ID]['Phase']['N_phase'] = {}
            FEATURES[patient_ID]['Phase']['A_phase'] = {}
            FEATURES[patient_ID]['Phase']['V_phase'] = {}

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
                crop_annotations( 'N_phase', patient_ID, N_folder, N_images[0], N_labels[0] )

            #if A_phase:
            A_images = glob.glob( os.path.join(CTimagefolder, f"{patient_ID}_*A.nrrd") )
            A_labels = glob.glob( os.path.join(CTlabelsfolder, f"{patient_ID}_*A.seg.nrrd") )
            A_folder = os.path.join(patientfolder, 'A_phase')
            if ( A_phase and len(A_images) > 0 and len(A_labels) > 0 ):
                try:
                    os.mkdir( A_folder )
                except:
                    pass
                crop_annotations( 'A_phase', patient_ID, A_folder, A_images[0], A_labels[0] )

            #if V_phase:
            V_images = glob.glob( os.path.join(CTimagefolder, f"{patient_ID}_*V.nrrd") )
            V_labels = glob.glob( os.path.join(CTlabelsfolder, f"{patient_ID}_*V.seg.nrrd") )
            V_folder = os.path.join(patientfolder, 'V_phase')
            if ( V_phase and len(V_images) > 0 and len(V_labels) > 0 ):
                try:
                    os.mkdir( V_folder )
                except:
                    pass
                crop_annotations( 'V_phase', patient_ID, V_folder, V_images[0], V_labels[0] )

        json_path = os.path.join( FEATURES_TABLE_OUTPUT, 'Features.json' )
        json_file = open(json_path, 'w')
        json_content = json.dumps(FEATURES, indent = 4)
        json_file.writelines(json_content)
        json_file.close()

        save_csv()
