from utility import TOOLBOX
from Model import SNeL_RFS_ANNC

import argparse
import os
import configparser
import numpy as np
import json

import time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

API_Description = """
***** Radiomics Analysis Platform  *****
API Name: TRAIN Model
Version:    1.0
Developer: Pei-Yan Li
Email:     d05548014@ntu.edu.tw
****************************************

"""

_pwd_ = os.getcwd()

parser = argparse.ArgumentParser(prog = 'Train.py',
                                 formatter_class = argparse.RawDescriptionHelpFormatter,
                                 description = API_Description)

parser.add_argument('-f', action = 'store', type = str, help='the path to the dataset stored in npz file format')
parser.add_argument('-hyp', action = 'store', type = str, help='the path to the hyper-parameter file')
parser.add_argument('-cfg', action = 'store', type = str, help='the architecture of neural netowrk designed for model')
parser.add_argument('-k', action = 'store_true', default=False, help='True/False to use 5-Fold cross validation for model training with 7:1:2 train-validation-test ratio.')
parser.add_argument('-s', action = 'store_true', default=False, help='True/False to standardize train data')
parser.add_argument('-p', action = 'store_true', default=False, help='True/False to use imputation algorithm for missing data')
parser.add_argument('-d', action = 'store', type = str, help='the folder for the storage of model checkpoints.')
parser.add_argument('-gpu', action = 'store', type = int, default = 0, help='the gpu id for training')
args = parser.parse_args()

if __name__ == '__main__':

    assert ( os.path.isfile(args.f) and ".npz" == os.path.splitext(args.f)[1] ), "Wrong! Given dataset cannot be found (*.npz)."
    assert ( os.path.isfile(args.hyp) and ".ini" == os.path.splitext(args.hyp)[1] ), "Wrong! Given hyper-parameter file cannot be found (*.ini)."
    assert ( os.path.isfile(args.cfg) and ".cfg" == os.path.splitext(args.cfg)[1] ), "Wrong! Given NN architecture file cannot be found (*.cfg)."

    if ( not os.path.isdir(args.d) ):
        os.mkdir(args.d)
    
    hyperparameters = configparser.ConfigParser()
    hyperparameters.read(args.hyp)

    nn_cfg = configparser.ConfigParser()
    nn_cfg.read(args.cfg)

    toolbox = TOOLBOX(kfold=args.k, normalize=args.s, file=args.f, hype=hyperparameters)

    if (args.k):
        print("Applying 5-fold cross validation for model training...")
        NNs = []
        for i in range(5):
            save_folder = os.path.join( args.d, f"Fold_{i+1}" )
            if ( not os.path.isdir(save_folder) ):
                os.mkdir(save_folder)
            NNs.append(
                SNeL_RFS_ANNC(cfg=nn_cfg, 
                        store=save_folder, gpu=int(args.gpu),
                        batchsize=toolbox.batch_size,
                        learning_rate=toolbox.learning_rate,
                        ID=i)
            )
        
        save_folder = os.path.join( args.d, f"Fold_Best" )
        if ( not os.path.isdir(save_folder) ):
            os.mkdir(save_folder)
        NNs.append(
                SNeL_RFS_ANNC(cfg=nn_cfg, 
                        store=save_folder, gpu=int(args.gpu),
                        batchsize=toolbox.batch_size,
                        learning_rate=toolbox.learning_rate,
                        ID=5)
            )

    Best_Set = {"Train":[], "Val":[], "Test":[]}
    Best_F1  = 0.0

    Fold_Count = 0
    CV_table = {}
    for TRAIN, VAL, TEST in toolbox.data_splitter():

        Train_X, Train_y = TRAIN
        Val_X, Val_y = VAL
        Test_X, Test_y = TEST

        if ( args.p ):
            Train_X, Val_X, Test_X = toolbox.Imputer_KNN()
        else:
            print("Replacing nans to zeros")
            Train_X, Val_X, Test_X = toolbox.fillzeros()

        if ( args.s ):
            print("Standardizing ...")
            Train_X, Val_X, Test_X = toolbox.Standard_Scaler()

        if (  0 < np.count_nonzero(np.isnan(Train_X))
           or 0 < np.count_nonzero(np.isnan(Val_X)) 
           or 0 < np.count_nonzero(np.isnan(Test_X))  ):
            print("NaN!!!")
            print(f"Train_X {np.count_nonzero(np.isnan(Train_X))}")
            print(f"Val_X {np.count_nonzero(np.isnan(Val_X))}")
            print(f"Test_X {np.count_nonzero(np.isnan(Test_X))}")
            exit()

        NNs[0].initModel()
        NNs[0].init_variables()
        
        EPOCHs = toolbox.get_final_epochs() - 1
        epoch_count = 0
        best_val_f1 = 0.0
        for lambda_s, lambda_a in toolbox.KKTmultipliers():
            
            print(f"CV-iteration: {Fold_Count+1}, Epoch: {epoch_count}")

            NNs[0].set_KKTmultipliers(lambda_s=lambda_s, lambda_a=lambda_a)

            start_time = time.time()

            loss = 0
            for i in range( Train_y.size ):
                x = Train_X[i, :]
                y = Train_y[i, :]

                try:
                    loss_,loss_softmax_ = NNs[0].run_train( x=np.reshape(x, (toolbox.batch_size, int(x.size / toolbox.batch_size)) ),
                                        label=y+1,
                                        step = epoch_count )
                    loss += loss_
                    print(f"epoch {epoch_count}, iter {i}, loss = {loss_}, loss_softmax={loss_softmax_}")
                except Exception as err:
                    print(f"Input x: {x}")
                    print(f"Input y: {y}")
                    print(err)

            train_loss = loss / Train_y.size

            best_val_f1, report = NNs[0].run_eval(X=Val_X, Y=Val_y, best_val_f1=best_val_f1, step=epoch_count)
        
            epoch_time = time.time() - start_time
            print("Epoch " + str(epoch_count) + " of " + str(EPOCHs) + " took " + str(epoch_time) + "s")
            print(f"Avergae training loss: {train_loss}")
            print(f"Best Validation Performance (F1): {best_val_f1}")

            epoch_count+=1
        
        CV_table[Fold_Count] = {'report':report, 'best f1':best_val_f1}
        
        if ( best_val_f1 > Best_F1 ):
            Best_F1 = best_val_f1
            Best_Set['Train'] = [Train_X, Train_y]
            Best_Set['Val']   = [Val_X, Val_y]
            Best_Set['Test']  = [Test_X, Test_y]

        del NNs[0]
        Fold_Count+=1
    
    json_path = os.path.join( args.d, 'CV_table.json' )
    json_file = open(json_path, 'w')
    json_content = json.dump( CV_table, indent=4 )
    json_file.writelines(json_content)
    json_file.close()

    NNs[0].initModel()
    NNs[0].init_variables()
    
    EPOCHs = toolbox.get_final_epochs()
    epoch_count = 0
    best_val_f1 = 0.0

    Train_X, Train_y = Best_Set['Train']
    Val_X, Val_y     = Best_Set['Val']
    Test_X, Test_y  = Best_Set['Test']

    Train_X = np.concatenate((Train_X, Val_X), axis=0)
    Train_y = np.concatenate((Train_y, Val_y), axis=0)

    best_report = None
    Final_test_table = {}

    for lambda_s, lambda_a in toolbox.KKTmultipliers():

        NNs[0].set_KKTmultipliers(lambda_s=lambda_s, lambda_a=lambda_a)

        start_time = time.time()

        loss = 0
        for i in range( Train_y.size ):
            x = Train_X[i, :]
            y = Train_y[i, :]

            loss_,loss_softmax_ = NNs[0].run_train( x=np.reshape(x, (toolbox.batch_size, int(x.size / toolbox.batch_size)) ),
                                    label=y+1,
                                    step = epoch_count )
            loss += loss_
            print(f"epoch {epoch_count}, iter {i}, loss = {loss_}, loss_softmax={loss_softmax_}, KKT_omega_s={KKT_omega_s_}, KKT_omega_a={KKT_omega_a_}")
        train_loss = loss / Train_y.size

        best_val_f1, best_report = NNs[0].run_eval(X=Test_X, Y=Test_y,
                                            best_val_f1=best_val_f1, best_report=best_report, 
                                            step=epoch_count)
    
        epoch_time = time.time() - start_time
        print("Epoch " + str(epoch_count) + " of " + str(EPOCHs) + " took " + str(epoch_time) + "s")

        Final_test_table['report'] = report
        Final_test_table['best f1'] = best_val_f1

        epoch_count+=1

    json_path = os.path.join( args.d, 'Final_test_table.json' )
    json_file = open(json_path, 'w')
    json_content = json.dump( Final_test_table, indent=4 )
    json_file.writelines(json_content)
    json_file.close()
    
    print("\n")
    print(f"Best Test Performance (F1): {best_val_f1}")
    print(best_report)


            



        

    
