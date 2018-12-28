import os
print(os.getcwd())
# os.chadir(r'C:\Users\31236\Desktop\baseline\LAL\data')
os.chdir(r'C:\Users\31236\Desktop\baseline\LAL\data')
print(os.getcwd())

import numpy as np
import scipy
import scipy.io as sio
from numpy import genfromtxt

# filename = r'rotated_checkerboard2x2_train.npz'
# dt = np.load(filename)

# trainData = dt['x']
# trainLabels = dt['y']

# print(np.shape(trainData))
# print(np.shape(trainLabels))
# print(trainData[0:3])
# print(trainLabels[0:5])
     


# filename = 'rotated_checkerboard2x2_test.npz'
# dt = np.load(filename)
# testData = dt['x']
# testLabels = dt['y']
# print(np.shape(testData))
# print(np.shape(testLabels))

# filename2 = 'striatum_test_features_mini.mat'
# dt = sio.loadmat(filename2)
# for i in dt.keys():
#     print(i)
#     print(np.shape(dt[i]))

# filename3 = 'striatum_test_labels_mini.mat'
# dt = sio.loadmat(filename3)
# for i in dt.keys():
#     print(i)
#     print(np.shape(dt[i]))
# print(dt.keys())
# # trainData = dt['features']
# trainLabels = dt['labels']

# # print(np.shape(trainData))
# print(np.shape(trainLabels))
# print(trainLabels[0:1])

# f1 = 'train.csv'
# f2 = 'test.csv'

# data = genfromtxt(f2, delimiter=',', dtype=np.str)
# index_name = np.array(data[0, 1:-1])
# print(np.shape(data))
# # train_label = data[1:,55].astype(np.int)

# traindata = data[1:,1:].astype(np.int)
# n_samples, n_feartures = np.shape(traindata)
# print(np.shape(traindata))
# print(traindata[0:5, 10])
# # merge wilderness_Area fearture
# wilderness_Area = np.zeros((n_samples,1), dtype=np.int)
# w_a = np.zeros(n_samples)
# for i in range(10,14):
#     wa = np.where(traindata[:, i]==1)[0]
#     wilderness_Area[wa] = i - 9
# print(wilderness_Area[0:5])
# # merge Soil_Type
# Soil_Type = np.zeros((n_samples,1), dtype=np.int)

# for round in range(14,54):
#     st = np.where(traindata[:, round]==1)[0]
#     Soil_Type[st] = round - 13

# # update the train data
# traindata = traindata[:, 0:10]
# traindata = np.concatenate((traindata, wilderness_Area, Soil_Type), axis=1)
# print(np.shape(traindata))
# # traindata = np.hstack((traindata, wilderness_Area, Soil_Type))
# index_name = index_name[0:10]
# index_name = np.concatenate((index_name, ['Wilderness_Area', 'Soil_Type']), axis=0)
# print(index_name)
# np.savez('Forest_Cover_Type_test.npz', testdata=traindata, indexname=index_name)

# f1 = 'Forest_Cover_Type_train.npz'
# data = np.load(f1)
# print(data.keys())
# traindata = data['traindata']
# trainlabel = data['trainlabel']
# print(np.shape(traindata))
# print(traindata[0])
# print(np.shape(trainlabel))
# print(trainlabel[0])

filename = './binary_classification/waveform-5000_1_2.csv'
data = genfromtxt(filename, delimiter=',', dtype=np.str)

print(np.shape(data))

traindata = data[0:550, 0:40].astype(np.float)
trainlabel = [data[0:550, 40].astype(np.float)]
trainlabel = np.transpose(trainlabel)
trainlabel[trainlabel==-1] = 0
testdata = data[550: , 0:8].astype(np.float)
testlabel = data[550: , 8].astype(np.float)
print(np.shape(traindata))
print(np.shape(trainlabel))
print(traindata[0:3])
print(trainlabel[0:5])
# print(np.shape(testdata))
# print(np.shape(testlabel))
# print(testdata[0])
# print(testlabel[0])