from ImputateMissingData import ImputationKNN
from sklearn.model_selection import KFold
import numpy as np

npzfile = "/media/share/DATA/Radiomics/Adrenal/data/output_features/RawFeatureData.npz"
kf = KFold(n_splits=10)

data = np.load(npzfile)
X = data["ARRAY"]
del data

print(f" Number of Missing Data in X = {np.count_nonzero(np.isnan(X))} ")

M, N = X.shape
SizeOfOneFold = int(M / 10)

SizeOfTrain = 7 * SizeOfOneFold
SizeOfValidate = 1 * SizeOfOneFold
SizeOfTest = 2 * SizeOfOneFold

for train_and_val_indices, validate_sampe_indice in kf.split(X):

    test_sample_indices = train_and_val_indices[0:SizeOfTest]
    train_sample_indice = train_and_val_indices[SizeOfTest::]

    train = X[train_sample_indice,:]
    val   = X[validate_sampe_indice, :]
    test  = X[test_sample_indices, :]

print(f" Number of Missing Data in train {train.shape} = {np.count_nonzero(np.isnan(train))}")
print(f" Number of Missing Data in val {val.shape}  = {np.count_nonzero(np.isnan(val))}")
print(f" Number of Missing Data in test {test.shape} = {np.count_nonzero(np.isnan(test))}")

imputer = ImputationKNN(train=train, val=val, test=test)

imputated_train, imputated_val, imputated_test = imputer.imputated_data()

print(f" Number of Missing Data in imputated_train {imputated_train.shape} = {np.count_nonzero(np.isnan(imputated_train))}")
print(f" Number of Missing Data in val {imputated_val.shape}  = {np.count_nonzero(np.isnan(imputated_val))}")
print(f" Number of Missing Data in imputated_test {imputated_test.shape} = {np.count_nonzero(np.isnan(imputated_test))}")
