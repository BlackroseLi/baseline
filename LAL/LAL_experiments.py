import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# import various AL strategies
from LAL.Classes.active_learner import ActiveLearnerRandom
from LAL.Classes.active_learner import ActiveLearnerUncertainty
from LAL.Classes.active_learner import ActiveLearnerLAL
from LAL.Classes.active_learner import ActiveLearnerSGD
# import the dataset class
from LAL.Classes.dataset import DatasetCheckerboard2x2
from LAL.Classes.dataset import DatasetCheckerboard4x4
from LAL.Classes.dataset import DatasetRotatedCheckerboard2x2
from LAL.Classes.dataset import DatasetStriatumMini
# import the model for LAL strategy
from LAL.Classes.lal_model import LALmodel
# import Experiment and Result classes that will be responsible for running AL and saving the results
from LAL.Classes.experiment import Experiment
from LAL.Classes.results import Results

from sklearn.linear_model import SGDRegressor
import random

# ------------------Build classifiers for LAL strategies----------------------------
# LALindependent strategy

fn = 'LAL-randomtree-simulatedunbalanced-big.npz'
# we found these parameters by cross-validating the regressor and now we reuse these expreiments
parameters = {'est': 2000, 'depth': 40, 'feat': 6 }
filename = './lal datasets/'+fn
regression_data1 = np.load(filename)
regression_features1 = regression_data1['arr_0']
regression_labels1 = regression_data1['arr_1']

print('Building lal regression model..')
lalModel1 = RandomForestRegressor(n_estimators = parameters['est'], max_depth = parameters['depth'], 
                                 max_features=parameters['feat'], oob_score=True, n_jobs=8)

lalModel1.fit(regression_features1, np.ravel(regression_labels1))    

print('Done!')
print('Oob score = ', lalModel1.oob_score_)

# LALiterative strategy

fn = 'LAL-iterativetree-simulatedunbalanced-big.npz'
# we found these parameters by cross-validating the regressor and now we reuse these expreiments
parameters = {'est': 1000, 'depth': 40, 'feat': 6 }
filename = './lal datasets/'+fn
regression_data2 = np.load(filename)
regression_features2 = regression_data2['arr_0']
regression_labels2 = regression_data2['arr_1']

print('Building lal regression model..')
lalModel2 = RandomForestRegressor(n_estimators = parameters['est'], max_depth = parameters['depth'], 
                                 max_features=parameters['feat'], oob_score=True, n_jobs=8)

lalModel2.fit(regression_features2, np.ravel(regression_labels2))    

print('Done!')
print('Oob score = ', lalModel2.oob_score_)

# SGD
print('Building sgd1 regression model..')
SGD1 = SGDRegressor(loss="squared_loss", penalty="l2", alpha=0.02, max_iter=100, random_state=random.randrange(100000))
SGD1.fit(regression_features1, regression_labels1)
print('Done!')

print('Building sgd2 regression model..')
SGD2 = SGDRegressor(loss="squared_loss", penalty="l2", alpha=0.02, max_iter=100, random_state=random.randrange(100000))
SGD2.fit(regression_features2, regression_labels2)
print('Done!')

## ----------------------Running the experiment: checkerboard 2x2---------------------

# number of experiment repeats
nExperiments = 20
# number of estimators (random trees) in the classifier
nEstimators = 50
# number of labeled points at the beginning of the AL experiment
nStart = 2
# number of iterations in AL experiment
nIterations = 100
# the quality metrics computed on the test set to evaluate active learners
quality_metrics = ['accuracy']

# load dataset
dtstcheckerboard2x2 = DatasetCheckerboard2x2()
# set the starting point
dtstcheckerboard2x2.setStartState(nStart)
# Active learning strategies
alR = ActiveLearnerRandom(dtstcheckerboard2x2, nEstimators, 'random')
alU = ActiveLearnerUncertainty(dtstcheckerboard2x2, nEstimators, 'uncertainty')
# lalModel1  lal-independent 
alLALindepend = ActiveLearnerLAL(dtstcheckerboard2x2, nEstimators, 'lal-rand', lalModel1)
# lalModel2 lal-interative
alLALiterative = ActiveLearnerLAL(dtstcheckerboard2x2, nEstimators, 'lal-iter', lalModel2)
# SGD
SGD1independ = ActiveLearnerSGD(dtstcheckerboard2x2, nEstimators,'SGD1independ', SGD1)
SGD1iterative = ActiveLearnerSGD(dtstcheckerboard2x2, nEstimators, 'SGD2iterative', SGD2)
als = [alR, alU, alLALindepend, alLALiterative, SGD1independ, SGD1iterative]

exp = Experiment(nIterations, nEstimators, quality_metrics, dtstcheckerboard2x2, als, 'here we can put a comment about the current experiments')
# the Results class helps to add, save and plot results of the experiments
res = Results(exp, nExperiments)

for i in range(nExperiments):
    print('\n experiment #'+str(i+1))
    # run an experiment
    performance = exp.run()
    res.addPerformance(performance)
    # reset the experiment (including sampling a new starting state for the dataset)
    exp.reset()

print()
res.saveResults('checkerboard2x2-exp')
print('checkerboard2x2-exp done')
# res2plot = Results()
# res2plot.readResult('checkerboard2x2-exp')
# res2plot.plotResults(metrics = ['accuracy'])

# ---------------------Running the experiment: checkerboard 4x4-------------------------------

# load dataset
dtstCheckerboard4x4 = DatasetCheckerboard4x4()
# other possible datasets: dtst = DatasetCheckerboard4x4(), dtst = DatasetRotatedCheckerboard2x2(), dtst = DatasetStriatumMini()
# set the starting point
dtstCheckerboard4x4.setStartState(nStart)
# Active learning strategies
alR = ActiveLearnerRandom(dtstCheckerboard4x4, nEstimators, 'random')
alU = ActiveLearnerUncertainty(dtstCheckerboard4x4, nEstimators, 'uncertainty')
alLALindepend = ActiveLearnerLAL(dtstCheckerboard4x4, nEstimators, 'lal-rand', lalModel1)
alLALiterative = ActiveLearnerLAL(dtstCheckerboard4x4, nEstimators, 'lal-iter', lalModel2)
als = [alR, alU, alLALindepend, alLALiterative]

exp = Experiment(nIterations, nEstimators, quality_metrics, dtstCheckerboard4x4, als, 'here we can put a comment about the current experiments')
# the Results class helps to add, save and plot results of the experiments
res = Results(exp, nExperiments)

for i in range(nExperiments):
    print('\n experiment #'+str(i+1))
    # run an experiment
    performance = exp.run()
    res.addPerformance(performance)
    # reset the experiment (including sampling a new starting state for the dataset)
    exp.reset()

print()    
res.saveResults('checkerboard4x4-exp')
print('checkerboard4x4-exp done')
# res2plot = Results()
# res2plot.readResult('checkerboard4x4-exp')
# res2plot.plotResults(metrics = ['accuracy'])

# -----------------------------Running the experiment: rotated checkerboard 2x2-----------------------------------------


# load dataset
dtstRotatedCheckerboard2x2 = DatasetRotatedCheckerboard2x2()
# other possible datasets: dtst = DatasetCheckerboard4x4(), dtst = DatasetRotatedCheckerboard2x2(), dtst = DatasetStriatumMini()
# set the starting point
dtstRotatedCheckerboard2x2.setStartState(nStart)
# Active learning strategies
alR = ActiveLearnerRandom(dtstRotatedCheckerboard2x2, nEstimators, 'random')
alU = ActiveLearnerUncertainty(dtstRotatedCheckerboard2x2, nEstimators, 'uncertainty')
alLALindepend = ActiveLearnerLAL(dtstRotatedCheckerboard2x2, nEstimators, 'lal-rand', lalModel1)
alLALiterative = ActiveLearnerLAL(dtstRotatedCheckerboard2x2, nEstimators, 'lal-iter', lalModel2)
als = [alR, alU, alLALindepend, alLALiterative]

exp = Experiment(nIterations, nEstimators, quality_metrics, dtstRotatedCheckerboard2x2, als, 'here we can put a comment about the current experiments')
# the Results class helps to add, save and plot results of the experiments
res = Results(exp, nExperiments)

for i in range(nExperiments):
    print('\n experiment #'+str(i+1))
    # run an experiment
    performance = exp.run()
    res.addPerformance(performance)
    # reset the experiment (including sampling a new starting state for the dataset)
    exp.reset()

print()
res.saveResults('rotated-checkerboard2x2-exp')
print('rotated-checkerboard2x2-exp done')
# res2plot = Results()
# res2plot.readResult('rotated-checkerboard2x2-exp')
# res2plot.plotResults(metrics = ['accuracy'])

# -----------------------------Running the experiment: mini striatum dataset------------------------------------------------

# load dataset
dtstStriatumMini = DatasetStriatumMini()
# set the starting point
dtstStriatumMini.setStartState(nStart)
# Active learning strategies
alR = ActiveLearnerRandom(dtstStriatumMini, nEstimators, 'random')
alU = ActiveLearnerUncertainty(dtstStriatumMini, nEstimators, 'uncertainty')
alLALindepend = ActiveLearnerLAL(dtstStriatumMini, nEstimators, 'lal-rand', lalModel1)
alLALiterative = ActiveLearnerLAL(dtstStriatumMini, nEstimators, 'lal-iter', lalModel2)
als = [alR, alU, alLALindepend, alLALiterative]

exp = Experiment(nIterations, nEstimators, quality_metrics, dtstStriatumMini, als, 'here we can put a comment about the current experiments')
# the Results class helps to add, save and plot results of the experiments
res = Results(exp, nExperiments)

for i in range(nExperiments):
    print('\n experiment #'+str(i+1))
    # run an experiment
    performance = exp.run()
    res.addPerformance(performance)
    # reset the experiment (including sampling a new starting state for the dataset)
    exp.reset()

print()
res.saveResults('striatum-exp')
print('striatum-exp done')
# res2plot = Results()
# res2plot.readResult('striatum-exp')
# res2plot.plotResults(metrics = ['IoU'])
