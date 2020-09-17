import numpy as np
import scipy as sp
import scipy.sparse as sps
import openml
import random
import pandas as pd
import sys
import os
from preprocessing import pre_process
from sklearn.preprocessing import LabelEncoder

#apikey_cy = '7bb5c2095921e391eb8b5f6b0ec9da51'
#openml.config.apikey = apikey_cy

i = sys.argv[1]

output_directory = 'preprocessed_datasets'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

selected_datasets = pd.read_csv("selected_OpenML_classification_dataset_indices.csv", index_col=None, header=None).values.T[0]
dataset_id = int(selected_datasets[int(i)])
print(dataset_id)

try:
    dataset = openml.datasets.get_dataset(dataset_id)
    data_features, data_labels, data_categorical, _ = dataset.get_data(target=dataset.default_target_attribute)
except:
    directory = '{}/datasets_with_error'.format(output_directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.system('touch '+str(directory)+'/'+str(dataset_id)+'.txt')

if type(data_features) is pd.core.frame.DataFrame or type(data_features) is pd.core.sparse.frame.SparseDataFrame:
    data_features = data_features.values
    
if sps.issparse(data_features):
    data_features=data_features.todense()

#     doing imputation and standardization and not doing one-hot-encoding achieves optimal empirical performances (smallest classification error) on a bunch of OpenML datasets
data_features, categorical = pre_process(raw_data=data_features, categorical=data_categorical, impute=True, standardize=True, one_hot_encode=True)

#     the output is a preprocessed dataset with all the columns except the last one being preprocessed features, and the last column being labels
# data = np.hstack(data_features_preprocessed, np.array(data_labels, ndmin=2).T)

pd.DataFrame(data_features, index=None, columns=None).to_csv(str(output_directory)+'/dataset_'+str(dataset_id)+'_features.csv', header=False, index=False)

le = LabelEncoder()

data_labels =le.fit_transform(data_labels)
pd.DataFrame(data_labels.reshape(-1, 1), index=None, columns=None).to_csv(str(output_directory)+'/dataset_'+str(dataset_id)+'_labels.csv', header=False, index=False)
                                
pd.DataFrame(np.array(data_categorical).reshape(-1, 1), index=None, columns=None).to_csv(str(output_directory)+'/dataset_'+str(dataset_id)+'_categorical.csv', header=False, index=False)

print("dataset "+str(dataset_id)+" finished")