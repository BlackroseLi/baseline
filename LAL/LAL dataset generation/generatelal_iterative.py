import numpy as np
import pickle as pkl
import scipy

# %matplotlib inline
import matplotlib.pyplot as plt

from Dataset4LAL import DatasetSimulated
from Tree4LAL import Tree4LAL
from LALmodel import LALmodel

experiment = dict()
# number of datasets for which we will generate data
experiment['n_datasets'] = 500
# how many datapoints will be labelled at the beginning, including 1 positive and 1 negative
experiment['n_labelleds'] = np.arange(2,50,1)
# how many times we will sample data with the same parameters
experiment['n_points_per_experiment'] = 10
# dimensionality of the data
experiment['n_dim'] = 2
# measure of quality change
experiment['method'] = 'error'
# for now 2 techniques for tree growing are available, random that means just adding random samples and iterative for adding points based on previously build model
experiment['treegrowing'] = 'iterative'

np.random.seed(805)

nDatapoints = 400
lalModels = []

all_data_for_lal = np.array([[]])
all_labels_for_lal = np.array([[]])

all_sizes_data_for_lal = np.array([[]])
all_sizes_labels_for_lal = np.array([[]])


for n_labelled in experiment['n_labelleds']:
    
    print()
    print('n_lablled = ', n_labelled)

    all_data_for_lal = np.array([[]])
    all_labels_for_lal = np.array([[]])
    for i_dataset in range(experiment['n_datasets']):
        print('*', end='')
        dataset = DatasetSimulated(nDatapoints, experiment['n_dim'])
        tree = Tree4LAL(experiment['treegrowing'], dataset, lalModels, experiment['method'])
        tree.generateTree(n_labelled)
        data_for_lal, labels_for_lal = tree.getLALdatapoints(experiment['n_points_per_experiment'])

        # stack LAL data together
        if np.size(all_data_for_lal)==0:
            all_data_for_lal = data_for_lal
            all_labels_for_lal = labels_for_lal
        else:
            all_data_for_lal = np.concatenate((all_data_for_lal, data_for_lal), axis=0)
            all_labels_for_lal = np.concatenate((all_labels_for_lal, labels_for_lal), axis=0)

    
    if experiment['treegrowing']=='iterative':
        # for every size of the tree train a lal model and attach it to the list of models for all sizes of trees
        # also let's do some cross validation to find better parameters 
        lalModel = LALmodel(all_data_for_lal, all_labels_for_lal)
        lalModel.crossValidateLALmodel()
        lalModels.append(lalModel.model)
    
    # data to save to build the big tree at the end
    
    if np.size(all_sizes_data_for_lal)==0:
        all_sizes_data_for_lal = all_data_for_lal
        all_sizes_labels_for_lal = all_labels_for_lal
    else:
        all_sizes_data_for_lal = np.concatenate((all_sizes_data_for_lal, all_data_for_lal), axis=0)
        all_sizes_labels_for_lal = np.concatenate((all_sizes_labels_for_lal, all_labels_for_lal), axis=0)
    np.savez('./lal datasets/LAL-iterativetree-simulated2Gauss2dim', all_sizes_data_for_lal, all_sizes_labels_for_lal)
    
lalModel = LALmodel(all_sizes_data_for_lal, all_sizes_labels_for_lal)
lalModel.crossValidateLALmodel()

print(all_sizes_data_for_lal.shape)
print(all_sizes_labels_for_lal.shape)

np.savez('./lal datasets/LAL-iterativetree-simulated2Gauss2dim', all_sizes_data_for_lal, all_sizes_labels_for_lal)