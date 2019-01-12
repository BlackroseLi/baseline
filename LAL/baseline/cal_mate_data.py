import copy
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import Normalizer,minmax_scale

from sklearn.datasets import make_classification
from sklearn.svm import SVC

from meta_data import DataSet, mate_data, mate_data_1, model_select

dataset_path = 'C:\\Users\\31236\\Desktop\\baseline\\data\\'
datasetnames = np.load('datasetname.npy')

dataset = DataSet('australian', dataset_path=dataset_path)
trains, tests, label_inds, unlabel_inds = dataset.split_data()
X = dataset.X
y = dataset.y
distacne = dataset.get_distance()
_, cluster_center_index = dataset.get_cluster_center()
N = 10
# print(np.shape(train))
# print(np.shape(test))

# # print(np.shape(train[0]))
# print(np.shape(l_ind))
# print(np.shape(u_ind))
# print(np.shape(l_ind[0]))
# print(l_ind[0])

# for name in datasetnames:
#     dataset = DataSet(name, dataset_path=dataset_path)
#     distacne = dataset.get_distance()
#     cluster_center_index = dataset.get_cluster_center()

#     
#     
    # test_ratio=0.3, initial_label_rate=0.05
    # for initial_l_r in range(0.03, 0.15, 0.01):
    #     train, test, l_ind, u_ind = dataset.split_data(test_ratio=0.3, initial_label_rate=initial_l_r, split_count=10)
    #     from sklearn.linear_model import LogisticRegression  
    #     model = LogisticRegression(penalty='l2')  
        
modelnames = ['KNN', 'LR']
metadata = []
perf_impr = []
for t in range(10):
    
    l_ind = label_inds[t]
    u_ind = unlabel_inds[t]
    test = tests[t]
    for modelname in modelnames:
        model = model_select(modelname, 1)

        modelOutput = []
        modelPerformance = []
        # genearte five rounds before
        labelindex = []
        unlabelindex = []
        for i in range(5):
            i_sampelindex = np.random.choice(u_ind)
            u_ind = np.delete(u_ind, np.where(u_ind == i_sampelindex)[0])
            l_ind = np.r_[l_ind, i_sampelindex]
            labelindex.append(l_ind)
            unlabelindex.append(u_ind)

            model_i = copy.deepcopy(model)
            model_i.fit(X[l_ind], y[l_ind].ravel())
            i_output = (model_i.predict_proba(X)[:, 1] - 0.5) * 2
            i_prediction = np.array([1 if k>0 else 0 for k in i_output])
            modelOutput.append(i_output)
            modelPerformance.append(accuracy_score(y[test], i_prediction[test]))
        
        for j in range(N):
            j_l_ind = copy.deepcopy(l_ind)
            j_u_ind = copy.deepcopy(u_ind)
            j_labelindex = copy.deepcopy(labelindex)
            j_unlabelindex = copy.deepcopy(unlabelindex)
            jmodelOutput = copy.deepcopy(modelOutput)

            j_sampelindex = np.random.choice(u_ind)
            j_u_ind = np.delete(j_u_ind, np.where(j_u_ind == j_sampelindex)[0])
            j_l_ind = np.r_[j_l_ind, j_sampelindex]
            labelindex.append(j_l_ind)
            unlabelindex.append(j_u_ind)

            model_j = copy.deepcopy(model)
            model_j.fit(X[j_l_ind], y[j_l_ind].ravel())
            j_output = (model_j.predict_proba(X)[:, 1] - 0.5) * 2
            jmodelOutput.append(j_output)
            j_prediction = np.array([1 if k>0 else 0 for k in j_output])
            j_meta_data = mate_data_1(X, y, distacne, cluster_center_index, labelindex, unlabelindex, jmodelOutput, j_sampelindex)
            metadata.append(j_meta_data)
            j_perf = accuracy_score(y[test], j_prediction[test])
            perf_impr.append(j_perf - modelPerformance[4])
        print(modelPerformance)

print(np.shape(metadata))
print(np.shape(perf_impr))
# print(metadata[0])
print(perf_impr[0:100])



