from ImputateMissingData import ImputationKNN

import numpy as np
import os
import configparser

#from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class TOOLBOX():

    def __init__(self, kfold = True, normalize = True, file=None, hype=None):
        
        self.kfold = kfold # True / False
        self.data = None
        self.keys = []
        self.normalize = normalize
        self.file = file
        self.hype = hype
        self.batch_size = self.hype.getint('Basic','batch_size')
        self.learning_rate = self.hype.getfloat('Basic','learning_rate')

        self.data_loader(file)
    
    def data_loader(self, file=None):

        if ( file is None ):
            return 

        filetype = os.path.splitext(file)[1]

        if ( ".npz" == filetype ):
            self.data = data = np.load(file)
            self.keys = list(self.data)    # 0: X, 1: Y        

    def data_splitter(self):
        '''
        
        yield three tuples of Train, Val, and Test data consisting of X and y
        
        '''

        M, N = self.data[self.keys[0]].shape
        SizeOfOneFold = int(M / 10)

        SizeOfTrain = 7 * SizeOfOneFold
        SizeOfValidate = 1 * SizeOfOneFold
        SizeOfTest = 2 * SizeOfOneFold

        if ( self.kfold ) :

            #kf = KFold(n_splits=10)
            kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)

            X = self.data[self.keys[0]]
            y = self.data[self.keys[1]]

            for train_indices, validate_sampe_indice in kf.split(X,y):
              
                test_sample_indice  = train_indices[0:SizeOfTest]
                train_sample_indice = train_indices[SizeOfTest::]

                self.train_X = np.copy(X[train_sample_indice, :])
                self.val_X  = np.copy(X[validate_sampe_indice, :])
                self.test_X = np.copy(X[test_sample_indice, :])

                self.train_y = np.copy(y[train_sample_indice, :])
                self.val_y  = np.copy(y[validate_sampe_indice, :])
                self.test_y = np.copy(y[test_sample_indice, :])

                yield (self.train_X, self.train_y), (self.val_X, self.val_y), (self.test_X, self.test_y)

        else:

            test_sample_indice = np.array( [ i for i in range(SizeOfTest) ] )
            validate_sampe_indice = np.array( [ i for i in range(SizeOfTest, SizeOfTest+SizeOfValidate)] )
            train_sample_indice = np.array( [ i for i in range(SizeOfTest+SizeOfValidate, M)])

            X = self.data[self.keys[0]]
            y = self.data[self.keys[1]]

            self.train_X = np.copy(X[train_sample_indice, :])
            self.val_X = np.copy(X[validate_sampe_indice, :])
            self.test_X = np.copy(X[test_sample_indice, :])

            self.train_y = np.copy(y[train_sample_indice, :])
            self.val_y = np.copy(y[validate_sampe_indice, :])
            self.test_y = np.copy(y[test_sample_indice, :])

            yield (self.train_X, self.train_y), (self.val_X, self.val_y), (self.test_X, self.test_y)

    def Standard_Scaler(self):

        scaler = StandardScaler()
        self.train_X = scaler.fit_transform(self.train_X)
        self.val_X   = scaler.transform(self.val_X)
        self.test_X  = scaler.transform(self.test_X)
        return self.train_X, self.val_X, self.test_X
    
    def Imputer_KNN(self):

        imputer = ImputationKNN(train=self.train_X, val=self.val_X, test=self.test_X)
        self.train_X, self.val_X, self.test_X = imputer.imputated_data()
        return self.train_X, self.val_X, self.test_X
    
    def fillzeros(self):

        self.train_X = np.nan_to_num(self.train_X)
        self.val_X   = np.nan_to_num(self.val_X)
        self.test_X  = np.nan_to_num(self.test_X)
        
        return self.train_X, self.val_X, self.test_X
    
    def get_final_epochs(self):

        epochs_per_stage = self.hype.getint('Basic','epochs_per_stage')
        cycles_lambda_s = self.hype.getint('Objective Function', 'cycles_lambda_s')
        step_lambda_s   = self.hype.getint('Objective Function','step_lambda_s')
        cycles_lambda_a = self.hype.getint('Objective Function','cycles_lambda_a')
        step_lambda_a   = self.hype.getint('Objective Function','step_lambda_a')

        self.stages = epochs_per_stage * cycles_lambda_s * ( 2 * step_lambda_s ) * cycles_lambda_a * ( 2 * step_lambda_a )

        min_lambda_s = self.hype.getfloat('Objective Function','min_lambda_s')
        max_lambda_s = self.hype.getfloat('Objective Function','max_lambda_s')
        s_interval = ( max_lambda_s - min_lambda_s ) / (step_lambda_s-1)

        min_lambda_a = self.hype.getfloat('Objective Function','min_lambda_a')
        max_lambda_a = self.hype.getfloat('Objective Function','max_lambda_a')
        a_interval = ( max_lambda_a - min_lambda_a ) / (step_lambda_a-1)

        self.lambda_Ss = []
        self.lambda_As = []

        for i in range(cycles_lambda_s):
            for j in range(step_lambda_s):
                lambda_s = min_lambda_s + s_interval * j
                buf_arr = [lambda_s] * cycles_lambda_a * 2 * step_lambda_a * epochs_per_stage
                self.lambda_Ss += buf_arr
            for j in range(step_lambda_s):
                lambda_s = max_lambda_s - s_interval * j
                buf_arr = [lambda_s] * cycles_lambda_a * 2 * step_lambda_a * epochs_per_stage
                self.lambda_Ss += buf_arr

        for j in range(step_lambda_a):
            lambda_a = min_lambda_a + a_interval * j
            buf_arr = [lambda_a] * epochs_per_stage
            self.lambda_As += buf_arr
        for j in range(step_lambda_a):
            lambda_a = max_lambda_a - a_interval * j
            buf_arr = [lambda_a] * epochs_per_stage
            self.lambda_As += buf_arr

        self.lambda_As = self.lambda_As * cycles_lambda_a * step_lambda_s * 2

        return self.stages

    def KKTmultipliers(self):

        for stage in range(self.stages):
            yield self.lambda_Ss[stage], self.lambda_As[stage]




        
