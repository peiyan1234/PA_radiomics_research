import argparse
import os
import json
from tempfile import TemporaryFile

import numpy as np

API_Description = """
***** Radiomics Analysis Platform  *****
API Name: Convert features json file to one npy file
Version:    1.0
Developer: Pei-Yan Li
Email:     d05548014@ntu.edu.tw
****************************************

"""

_pwd_ = os.getcwd()

parser = argparse.ArgumentParser(prog = 'get_features_json2npz.py',
                                 formatter_class = argparse.RawDescriptionHelpFormatter,
                                 description = API_Description)

parser.add_argument('-f', action = 'store', type = str, help='the path to the Features.json')
args = parser.parse_args()

def readfile( file ):

    data = open( file,'r' );
    return json.load(data);

def converter( table ):

    ListOfPatients = list( table.keys() )
    
    NumberOfPatients = len( ListOfPatients )
    elimination = 0
    wait2removed = []
    for p in ListOfPatients:
        if ( not table[p]["Phase"]["N_phase"] and
             not table[p]["Phase"]["A_phase"] and
             not table[p]["Phase"]["V_phase"] or 
             "" == table[p]["lateralization"] ):
             elimination += 1
             wait2removed.append(p)
             print(f"Remove {p}")
    for p in wait2removed:
        ListOfPatients.remove(p)
    print(f"Removed {len(wait2removed)} invalid data")
    NumberOfPatients-= elimination

    NumberOfFeatures = len(list(table["PA1"]["Phase"]["N_phase"]["Adrenal"]["Original"].keys()))
    
    ARRAY = np.empty( (NumberOfPatients, 2*3*NumberOfFeatures), dtype = float )
    ARRAY[:] = np.nan

    Y = np.empty( (NumberOfPatients, 1), dtype = float )
    Y[:] = np.nan

    print(f" Number of Patients = {NumberOfPatients}")
    print(f" Number of Features = {NumberOfFeatures}")

    for i in range(NumberOfPatients):
        p = ListOfPatients[i]
        count = 0

        label = table[p]["lateralization"]
        if ( 'N' == label ):
            Y[i] = 0
        elif ( 'L' == label ):
            Y[i] = -1
        elif ( 'R' == label ):
            Y[i] = 1
        else:
            continue
        
        # Left Adrenal Features
        if ( 0 < len(list(table[p]["Phase"]["N_phase"].keys())) ):
            for type in table[p]["Phase"]["N_phase"]["LeftAdrenal"]["Original"]:
                ARRAY[ i, count ] = table[p]["Phase"]["N_phase"]["LeftAdrenal"]["Original"][type]
                count+=1
        else:
            count+=NumberOfFeatures

        if ( 0 < len(list(table[p]["Phase"]["A_phase"].keys())) ):
            for type in table[p]["Phase"]["A_phase"]["LeftAdrenal"]["Original"]:
                ARRAY[ i, count ] = table[p]["Phase"]["A_phase"]["LeftAdrenal"]["Original"][type]
                count+=1
        else:
            count+=NumberOfFeatures

        if ( 0 < len(list(table[p]["Phase"]["V_phase"].keys())) ):
            for type in table[p]["Phase"]["V_phase"]["LeftAdrenal"]["Original"]:
                ARRAY[ i, count ] = table[p]["Phase"]["V_phase"]["LeftAdrenal"]["Original"][type]
                count+=1
        else:
            count+=NumberOfFeatures
        
        # Right Adrenal Features
        if ( 0 < len(list(table[p]["Phase"]["N_phase"].keys())) ):
            for type in table[p]["Phase"]["N_phase"]["RightAdrenal"]["Original"]:
                ARRAY[ i, count ] = table[p]["Phase"]["N_phase"]["RightAdrenal"]["Original"][type]
                count+=1
        else:
            count+=NumberOfFeatures

        if ( 0 < len(list(table[p]["Phase"]["A_phase"].keys())) ):
            for type in table[p]["Phase"]["A_phase"]["RightAdrenal"]["Original"]:
                ARRAY[ i, count ] = table[p]["Phase"]["A_phase"]["RightAdrenal"]["Original"][type]
                count+=1
        else:
            count+=NumberOfFeatures

        if ( 0 < len(list(table[p]["Phase"]["V_phase"].keys())) ):
            for type in table[p]["Phase"]["V_phase"]["RightAdrenal"]["Original"]:
                ARRAY[ i, count ] = table[p]["Phase"]["V_phase"]["RightAdrenal"]["Original"][type]
                count+=1
        else:
            count+=NumberOfFeatures
    
    M, N = ARRAY.shape
    print(f" Shape of ARRAY = {ARRAY.shape}")
    print(f" Shape of Y = {Y.shape}")
    print(f" Number of Missing Data in ARRAY = {np.count_nonzero(np.isnan(ARRAY))}")
    print(f" Number of Missing Data in Y = {np.count_nonzero(np.isnan(Y))}")
    print(f" Ratio of Missing data { round(100 * np.count_nonzero(np.isnan(ARRAY)) / (M*N), 2) }%")

    outputfile = os.path.join( os.path.split(args.f)[0], 'RawFeatureData.npz')
    np.savez(outputfile, ARRAY=ARRAY, Y=Y)

if __name__ == '__main__':

    if ( os.path.isfile(args.f) ):
        try:
            table = readfile( args.f )
        except Exception as err:
            print(f"Failed to read {args.f}")
            raise err
        
        if ( table ):
            try:
                converter( table )
            except Exception as err:
                print(f"Failed to convert json table to npz and save")
                raise err