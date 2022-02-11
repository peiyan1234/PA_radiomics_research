import numpy as np
import math

class ImputationKNN():

    def __init__(self, train=None, val=None, test=None, y=None):
        
        self.train = None
        self.val   = None
        self.test  = None
        self.y     = None
        self.NumOfTrainSample = 0
        self.NumOfValSample   = 0
        self.NumOfTestSample  = 0
        self.NumOfVariables = 0
        self.K = 0  # generally choose the value of sqrt(self.NumberOfTrainSample)
                    # A rule of thumb in machine learning is to pick
                    # K is suggested to choose an odd value to avoid a tie

        self.imputated_train = None
        self.imputated_val  = None
        self.imputated_test = None

        if ( train is None or test is None ):
            return
        else:
            self.train = train
            self.y     = y
            self.val   = val
            self.test  = test
            M, N = train.shape
            m, n = test.shape
            self.NumOfTrainSample = M
            self.NumOfTestSample  = m
            if ( self.val is not None ):
                i, j = val.shape
                self.NumOfValSample = i
            self.NumOfVariables   = N
            self.K = int(math.sqrt(M))
            if ( 0 < self.K % 2 ):
                self.K += 1 # K is suggested to choose an odd value to avoid a tie
            
            print("Finding Missing Data")
            self.findout()

            print(f"Imputate Missing Data by KNN, of which k is {self.K}")
            self.imputer()

    def __del__(self):

        print("the instance of ImputationKNN was deleted.")

    def findout(self):

        self.MissingInTrain = []
        self.MissingInVal   = []
        self.MissingInTest  = []

        for i in range(self.NumOfTrainSample):
            if ( 0 < np.count_nonzero(np.isnan(self.train[i,:])) ):
                self.MissingInTrain.append(i)
        print(f"Find {len(self.MissingInTrain)} patients with missing data in train")

        for i in range(self.NumOfValSample):
            if ( 0 < np.count_nonzero(np.isnan(self.val[i, :])) ):
                self.MissingInVal.append(i)
        print(f"Find {len(self.MissingInVal)} patients with missing data in val")
        
        for j in range(self.NumOfTestSample):
            if ( 0 < np.count_nonzero(np.isnan(self.test[j, :])) ):
                self.MissingInTest.append(j)
        print(f"Find {len(self.MissingInTest)} patients with missing data in test")

    def imputer(self):

        self.imputated_train = self.train.copy() if ( self.train is not None ) else np.array([])
        self.imputated_test  = self.test.copy() if ( self.test is not None ) else np.array([])
        self.imputated_val   = self.val.copy() if ( self.val is not None ) else np.array([])

        self.denominator = np.empty( (self.NumOfVariables, 1), dtype = float )
        for i in range(self.NumOfVariables):
            n_max = self.train[0, i] if ( not np.isnan(self.train[0, i]) ) else np.finfo(self.train.dtype).min
            n_min = self.train[0, i] if ( not np.isnan(self.train[0, i]) ) else np.finfo(self.train.dtype).max
            for j in range(1, self.NumOfTrainSample):
                n_max = self.train[j, i] if ( self.train[j,i] > n_max ) else n_max
                n_min = self.train[j, i] if ( self.train[j,i] < n_min ) else n_min
            denominator = n_max - n_min
            self.denominator[i] = n_max - n_min if ( 0 != n_max-n_min ) else 1
        
        for j in self.MissingInTrain:
            for i in range(self.NumOfVariables):
                if ( np.isnan(self.train[j, i]) ):
                    neighbors = []
                    for k in range(self.NumOfTrainSample):
                        if ( k!=j and not np.isnan(self.train[k,i])):
                            neighbors.append(k)
                    distances = self.HEOM_distance(neighbors=neighbors, sample_arr=self.train[j,:], matrix=self.train)
                    count = 0
                    sum = 0
                    weight_sum = 0
                    for k in list(distances.keys()):
                        if ( count == self.K ):
                            break
                        weight_sum = weight_sum + 1 / (distances[k] * distances[k] )
                        if ( np.isnan(weight_sum) ):
                            print(f"nan occur, distance: {distances[k], k}")
                        sum = sum + (self.train[k, i]) / (distances[k] * distances[k])
                        count+=1
                    self.imputated_train[j, i] = float( sum / weight_sum ) if ( 0 < weight_sum ) else 0.0
                    print(f"imputing train[{j},{i}] with {self.imputated_train[j, i]}, weight_sum={weight_sum}")

        for j in self.MissingInTest:
            for i in range(self.NumOfVariables):
                if ( np.isnan(self.test[j, i]) ):
                    neighbors = []
                    for k in range(self.NumOfTrainSample):
                        if ( not np.isnan(self.train[k,i])):
                            neighbors.append(k)
                    distances = self.HEOM_distance(neighbors=neighbors, sample_arr=self.test[j,:], matrix=self.train)
                    count = 0
                    sum = 0
                    weight_sum = 0
                    for k in list(distances.keys()):
                        if ( count == self.K ):
                            break
                        weight_sum = weight_sum + 1 / (distances[k] * distances[k] )
                        sum = sum + (self.train[k, i]) / (distances[k] * distances[k])
                        count+=1
                    self.imputated_test[j, i] = float( sum / weight_sum ) if ( 0 < weight_sum ) else 0.0
                    print(f"imputing test[{j},{i}] with {self.imputated_test[j,i]}")

        for j in self.MissingInVal:
            for i in range(self.NumOfVariables):
                if ( np.isnan(self.val[j, i]) ):
                    neighbors = []
                    for k in range(self.NumOfTrainSample):
                        if ( not np.isnan(self.train[k,i])):
                            neighbors.append(k)
                    distances = self.HEOM_distance(neighbors=neighbors, sample_arr=self.val[j,:], matrix=self.train)
                    count = 0
                    sum = 0
                    weight_sum = 0
                    for k in list(distances.keys()):
                        if ( count == self.K ):
                            break
                        weight_sum = weight_sum + 1 / (distances[k] * distances[k] )
                        sum = sum + (self.train[k, i]) / (distances[k] * distances[k])
                        count+=1
                    self.imputated_val[j, i] = float( sum / weight_sum ) if ( 0 < weight_sum ) else 0.0
                    print(f"imputing val[{j},{i}] with {self.imputated_val[j,i]}")

    def HEOM_distance(self, neighbors = None, matrix = None, sample_arr = None):
        
        distance_table = {}

        a = sample_arr
        for j in neighbors:
            neighbor = matrix[j, :]
            dist = 0
            for i in range(self.NumOfVariables):
                if ( np.isnan(a[i]) or np.isnan(neighbor[i]) ):
                    dist += 1
                else:
                    d = abs( a[i] - neighbor[i] ) / self.denominator[i]
                    dist += ( d * d )
            dist = math.sqrt(dist)
            distance_table[j] = dist
        distances = sorted( distance_table.items(), key=lambda item:item[1] )
        return dict(distances)
    
    def imputated_data(self):

        return self.imputated_train, self.imputated_val, self.imputated_test
        
