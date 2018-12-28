import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle
import os
os.chdir(r'C:\Users\31236\Desktop\baseline\LAL')
# import various AL strategies
from Classes.active_learner import ActiveLearnerRandom
from Classes.active_learner import ActiveLearnerUncertainty
from Classes.active_learner import ActiveLearnerLAL
from Classes.active_learner import ActiveLearnerSGD
# import the dataset class
from Classes.dataset import DatasetCheckerboard2x2
from Classes.dataset import DatasetCheckerboard4x4
from Classes.dataset import DatasetRotatedCheckerboard2x2
from Classes.dataset import DatasetStriatumMini
from Classes.dataset import DatasetSimulatedUnbalanced, DatasetBreast, DatasetDiabetes, Datasetwaveform_5000_1_2
# import the model for LAL strategy
from Classes.lal_model import LALmodel
# import Experiment and Result classes that will be responsible for running AL and saving the results
from Classes.experiment import Experiment
from Classes.results import Results

from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
import random

fn = 'LAL-randomtree-simulatedunbalanced-big.npz'
# we found these parameters by cross-validating the regressor and now we reuse these expreiments
parameters = {'est': 2000, 'depth': 40, 'feat': 6 }
filename = './lal datasets/'+fn
regression_data1 = np.load(filename)
regression_features1 = regression_data1['arr_0']
regression_labels1 = regression_data1['arr_1']


fn = 'LAL-iterativetree-simulatedunbalanced-big.npz'
# we found these parameters by cross-validating the regressor and now we reuse these expreiments
parameters = {'est': 1000, 'depth': 40, 'feat': 6 }
filename = './lal datasets/'+fn
regression_data2 = np.load(filename)
regression_features2 = regression_data2['arr_0']
regression_labels2 = regression_data2['arr_1']

# with open('LALmodel1','rb') as f:
#     lalModel1 = pickle.load(f)

# with open('LALmodel2','rb') as f:
#     lalModel2 = pickle.load(f)

# SGD
print('Building sgd1 regression model..')
SGD1 = SGDRegressor(loss="squared_loss", penalty="l2", alpha=0.02, max_iter=100, random_state=random.randrange(100000))
SGD1.fit(regression_features1, regression_labels1)
print('Done!')

print('Building sgd2 regression model..')
SGD2 = SGDRegressor(loss="squared_loss", penalty="l2", alpha=0.02, max_iter=100, random_state=random.randrange(100000))
SGD2.fit(regression_features2, regression_labels2)
print('Done!')
# linearRegession
linearReg1 = LinearRegression().fit(regression_features1, regression_labels1)
linearReg2 = LinearRegression().fit(regression_features2, regression_labels2)
print('Build linearRegresion regression model..')

## ----------------------Running the experiment: checkerboard 2x2---------------------

# number of experiment repeats
nExperiments = 1
# number of estimators (random trees) in the classifier
nEstimators = 50
# number of labeled points at the beginning of the AL experiment
nStart = 100
# number of iterations in AL experiment
nIterations = 100
# the quality metrics computed on the test set to evaluate active learners
quality_metrics = ['accuracy']

# load dataset
dataset = Datasetwaveform_5000_1_2()
# set the starting point
dataset.setStartState(nStart)
# Active learning strategies
alR = ActiveLearnerRandom(dataset, nEstimators, 'random')
alU = ActiveLearnerUncertainty(dataset, nEstimators, 'uncertainty')
# alLALindepend = ActiveLearnerLAL(dataset, nEstimators, 'lal-rand', lalModel1)
# alLALiterative = ActiveLearnerLAL(dataset, nEstimators, 'lal-iter', lalModel2)
# SGD
SGD1independ = ActiveLearnerSGD(dataset, nEstimators,'SGD1independ', SGD1)
SGD1iterative = ActiveLearnerSGD(dataset, nEstimators, 'SGD2iterative', SGD2)

# LinearRegression
lr1 = ActiveLearnerSGD(dataset, nEstimators,'linearReg1', linearReg1)
lr2 = ActiveLearnerSGD(dataset, nEstimators,'linearReg2', linearReg2)


als = [alR, alU, SGD1independ, SGD1iterative, lr1, lr2]

exp = Experiment(nIterations, nEstimators, quality_metrics, dataset, als,'SGD_experiment')
# the Results class helps to add, save and plot results of the experiments
res = Results(exp, nExperiments)

for i in range(nExperiments):
    print('\n experiment #'+str(i+1))
    # run an experiment
    performance = exp.run()
    res.addPerformance(performance)
    # reset the experiment (including sampling a new starting state for the dataset)
    exp.reset()

res.saveResults('DatasetDiabetes-exp')
print('experiment results save done')
res.plotResults(metrics = ['accuracy'])

