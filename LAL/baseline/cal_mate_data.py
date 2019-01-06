import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer,minmax_scale

from sklearn.datasets import make_classification
from sklearn.svm import SVC

def cal_matedata(label_data, label_y, unlabel_data, modelPrediction, al5tbefor, query_index):

    """
    It assume the binary classification.
    Parameters
    ----------
    label_data: {list, np.ndarray}

    label_y = {list, np.ndarray}
        
    label_index: {list, np.ndarray}
        The indexes of labeled samples.

    unlabel_index: {list, np.ndarray}
        The indexes of unlabeled samples.
    labeldata: {list, np.ndarray}

    unlabeldata:

    modelPrediction:
        The current model predicts for the whole dataset,corresponding to labelset and unlabelset.

    model:
        model must has predict()  predict_proba()
    """
    assert(isinstance(label_data, (list, np.ndarray)))
    assert(isinstance(label_y, (list, np.ndarray)))
    assert(isinstance(unlabel_data, (list, np.ndarray)))
    assert(isinstance(modelPrediction, (list, np.ndarray)))

    label_size, n_features = np.shape(label_data)
    unlabel_size = np.shape(unlabel_data)[0]
    assert(n_features == np.shape(unlabel_data)[1])
    assert(label_size == np.shape(label_y)[0])

    # binary classification 
    assert(np.shape(label_y)[1] == 1)
    assert(len(np.unique(label_y)) == 2)


    ratio_label_positive = (sum(label_y > 0)) / label_size
    ratio_label_negative = (sum(label_y < 0)) / label_size

    ratio_unlabel_positive = (sum(modelPrediction[label_size:] > 0)) / unlabel_size
    ratio_unlabel_negative = (sum(modelPrediction[label_size:] < 0)) / unlabel_size

    label_cluster = KMeans(n_clusters=10).fit(label_data)
    label_cluster_centers_10 = label_cluster.cluster_centers_
    unlabel_cluster = KMeans(n_clusters=10).fit(unlabel_data)
    unlabel_cluster_centers_10 = unlabel_cluster.cluster_centers_

    sorted_label_data = np.argsort(modelPrediction[0:label_size])
    label_10_equal = np.array([[int(round(i * label_size))] for i in np.arange(0, 1, 0.1)])
    
    unlabel_10_equal = np.array([ np.argsort(modelPrediction[label_size:])[int(round(i * label_size))] for i in np.arange(0, 1, 0.1)])





    tn, fp, fn, tp = confusion_matrix(label_y, modelPrediction[0: label_size], labels=[-1, 1]).ravel()
    ratio_tn = tn / label_size
    ratio_fp = fp / label_size
    ratio_fn = fn / label_size
    ratio_tp = tp / label_size


def matedata(X, label_ys, label_indexs, unlabel_indexs, modelPredictions, query_index, currentmodel):
    """Calculate the meta data according to the current model,dataset and five rounds before information.


    Parameters
    ----------
    X: 2D array
        Feature matrix of the whole dataset. It is a reference which will not use additional memory.

    label_ys:  {list, np.ndarray}
        The true label of the each round of iteration,corresponding to label_indexs.

    label_indexs: {list, np.ndarray} shape=(number_iteration, corresponding_label_index)
        The label indexs of each round of iteration,

    unlabel_indexs: {list, np.ndarray} shape=(number_iteration, corresponding_unlabel_index)
        The unlabel indexs of each round of iteration,

    modelPredictions: {list, np.ndarray} shape=(number_iteration, corresponding_perdiction)


    query_index: {list, np.ndarray}
        The unlabel samples will be queride,and calculate the performance improvement after add to the labelset.

    Returns
    -------
    metadata: 2D array
        The meta data about the current model and dataset.
    """

    for i in range(5):
        assert(len(label_ys[i]) == len(label_indexs[i]))
        assert(np.shape(X)[0] == np.shape(modelPredictions[i])[0]) 
        if(not isinstance(label_indexs[i], np.ndarray)):
            label_indexs[i] = np.array(label_indexs[i])
        if(not isinstance(unlabel_indexs[i], np.ndarray)):
            unlabel_indexs[i] = np.array(unlabel_indexs[i])
    
    n_samples, n_feature = np.shape(X)
    query_index_size = len(query_index)
    n_feature_data = n_feature * np.ones((query_index_size, 1))
    current_label_data = X[label_indexs[5]]
    current_label_size = len(label_indexs[5])
    current_label_y = label_ys[5]
    current_unlabel_data = X[unlabel_indexs[5]]
    current_unlabel_size = len(unlabel_indexs[5])
    current_prediction = modelPredictions[5]

    ratio_label_positive = (sum(current_label_y > 0)) / current_label_size
    ratio_label_positive_data = ratio_label_positive * np.ones_like(n_feature_data)
    ratio_label_negative = (sum(current_label_y < 0)) / current_label_size
    ratio_label_negative_data = ratio_label_negative * np.ones_like(n_feature_data)

    ratio_unlabel_positive = (sum(current_prediction[unlabel_indexs[5]] > 0)) / current_unlabel_size
    ratio_unlabel_positive_data = ratio_unlabel_positive * np.ones_like(n_feature_data)
    ratio_unlabel_negative = (sum(current_prediction[unlabel_indexs[5]] < 0)) / current_unlabel_size
    ratio_unlabel_negative_data = ratio_unlabel_negative * np.ones_like(n_feature_data)

    
    # label_cluster = KMeans(n_clusters=10).fit(current_label_data)
    # label_cluster_centers_10 = label_cluster.cluster_centers_
    # label_cluster_centers_10_index = np.zeros(10, dtype=int)

    # unlabel_cluster = KMeans(n_clusters=10).fit(current_unlabel_data)
    # unlabel_cluster_centers_10 = unlabel_cluster.cluster_centers_
    # unlabel_cluster_centers_10_index = np.zeros(10, dtype=int)

    # the same dataset the same cluster centers
    data_cluster = KMeans(n_clusters=10).fit(X)
    data_cluster_centers_10 = data_cluster.cluster_centers_
    closest_distance_data_cluster_centers_10 = np.zeros(10) + np.infty
    data_cluster_centers_10_index = np.zeros(10, dtype=int) - 1

    # obtain the cluster centers index
    for i in range(n_samples):
        for j in range(10):
            # if(np.all(data_cluster_centers_10[j] == X[i])):
            #     data_cluster_centers_10_index[j] = i
            distance = np.linalg.norm(X[i] - data_cluster_centers_10[j])
            if distance < closest_distance_data_cluster_centers_10[j]:
                closest_distance_data_cluster_centers_10[j] = distance
                data_cluster_centers_10_index[j] = i

    print('data_cluster_centers_10_index', data_cluster_centers_10_index)
    data_cluster_centers_10 = X[data_cluster_centers_10_index]
    print('data_cluster_centers_10', data_cluster_centers_10)
    if(np.any(data_cluster_centers_10_index == -1)):
        raise IndexError("data_cluster_centers_10_index is wrong")
    
    
    sorted_labelperdiction_index = np.argsort(current_prediction[label_indexs[5]])
    sorted_current_label_data = X[label_indexs[5][sorted_labelperdiction_index]]
    
    label_10_equal = [sorted_current_label_data[int(i * current_label_size)] for i in np.arange(0, 1, 0.1)]
    label_10_equal_index = [label_indexs[5][sorted_labelperdiction_index][int(i * current_label_size)] for i in np.arange(0, 1, 0.1)]

    sorted_unlabelperdiction_index = np.argsort(current_prediction[unlabel_indexs[5]])
    sorted_current_unlabel_data = X[unlabel_indexs[5][sorted_unlabelperdiction_index]]
    # unlabel_10_equal_index = np.zeros(10, dtype=int)
    # print('current un label size', current_unlabel_size)
    # print(len())
    unlabel_10_equal = [sorted_current_unlabel_data[int(i * current_unlabel_size)] for i in np.arange(0, 1, 0.1)]
    unlabel_10_equal_index = [unlabel_indexs[5][sorted_unlabelperdiction_index][int(i * current_unlabel_size)] for i in np.arange(0, 1, 0.1)]

    # for i in range(n_samples):
    #     for j in range(10):
    #         if(np.all(label_cluster_centers_10[j] == X[i])):
    #             label_cluster_centers_10_index[j] = i
    #         if(np.all(unlabel_cluster_centers_10[j] == X[i])):
    #             unlabel_cluster_centers_10_index[j] = i
    #         # if(np.all(label_10_equal[j] == X[i])):
    #         #     label_10_equal_index[j] = i
    #         # if(np.all(label_10_equal[j] == X[i])):
    #         #     unlabel_10_equal_index[j] = i
              
    distance_query_data = None
    cc_sort_index = []
    # lcc_sort_index = []
    # ucc_sort_index = []
    for i in query_index:
        # i_lcc = []
        # i_ucc = []
        i_cc = []
        i_l10e = []
        i_u10e = []

        # f_i_lcc = []
        # f_i_ucc = []
        # f_i_l10e = []
        # f_i_u10e = []      
        for j in range(10):
            # cal the ith in query_index about 
            # i_lcc.append(np.linalg.norm(current_unlabel_data[i] - label_cluster_centers_10[j]))
            # i_ucc.append(np.linalg.norm(current_unlabel_data[i] - unlabel_cluster_centers_10[j]))
            # i_cc.append(np.linalg.norm(current_unlabel_data[i] - data_cluster_centers_10[j]))
            # i_l10e.append(np.linalg.norm(current_unlabel_data[i] - label_10_equal[j]))
            # i_u10e.append(np.linalg.norm(current_unlabel_data[i] - unlabel_10_equal[j]))
            i_cc.append(np.linalg.norm(X[i] - data_cluster_centers_10[j]))
            i_l10e.append(np.linalg.norm(X[i] - label_10_equal[j]))
            i_u10e.append(np.linalg.norm(X[i] - unlabel_10_equal[j]))
        
        # i_lcc = minmax_scale(i_lcc)
        # i_lcc_sort_index = np.argsort(i_lcc)
        # lcc_sort_index.append(i_lcc_sort_index)
        # # i_lcc = np.sort(i_lcc)
        # i_ucc = minmax_scale(i_ucc)
        # i_ucc_sort_index = np.argsort(i_ucc)
        # ucc_sort_index.append(i_ucc_sort_index)
        # i_ucc = np.sort(i_ucc)

        i_cc = minmax_scale(i_cc)
        i_cc_sort_index = np.argsort(i_cc)
        # cc_sort_index.append(data_cluster_centers_10_index[i_cc_sort_index])
        cc_sort_index.append(i_cc_sort_index)
        i_l10e = minmax_scale(i_l10e)
        i_u10e = minmax_scale(i_u10e)
        # i_distance = np.hstack((i_lcc[i_lcc_sort_index], i_ucc[i_ucc_sort_index], i_l10e, i_u10e))
        i_distance = np.hstack((i_cc[i_cc_sort_index], i_l10e, i_u10e))
        if distance_query_data is None:
            distance_query_data = i_distance
        else:
            distance_query_data = np.vstack((distance_query_data, i_distance))

    ratio_tn = []
    ratio_fp = []
    ratio_fn = []
    ratio_tp = []
    label_pre_10_equal = []
    labelmean = []
    labelstd = []
    unlabel_pre_10_equal = []
    round5_ratio_unlabel_positive = []
    round5_ratio_unlabel_negative = []
    unlabelmean = []
    unlabelstd = []   
    for i in range(6):
        label_size = len(label_indexs[i])
        unlabel_size = len(unlabel_indexs[i])
        cur_prediction = modelPredictions[i]
        label_ind = label_indexs[i]
        unlabel_ind = unlabel_indexs[i]

        tn, fp, fn, tp = confusion_matrix(label_ys[i], cur_prediction[label_ind], labels=[-1, 1]).ravel()
        ratio_tn.append(tn / label_size)
        ratio_fp.append(fp / label_size)
        ratio_fn.append(fn / label_size)
        ratio_tp.append(tp / label_size)

        sort_label_pred = np.sort(minmax_scale(cur_prediction[label_ind]))
        i_label_10_equal = [sort_label_pred[int(i * label_size)] for i in np.arange(0, 1, 0.1)]
        label_pre_10_equal = np.r_[label_pre_10_equal, i_label_10_equal]
        # label_pre_10_equal.append(i_label_10_equal)
        labelmean.append(np.mean(i_label_10_equal))
        labelstd.append(np.std(i_label_10_equal))

        round5_ratio_unlabel_positive.append((sum(current_prediction[unlabel_ind] > 0)) / unlabel_size)
        round5_ratio_unlabel_negative.append((sum(current_prediction[unlabel_ind] < 0)) / unlabel_size)
        sort_unlabel_pred = np.sort(minmax_scale(cur_prediction[unlabel_ind]))
        i_unlabel_10_equal = [sort_unlabel_pred[int(i * unlabel_size)] for i in np.arange(0, 1, 0.1)]
        # unlabel_pre_10_equal.append(i_unlabel_10_equal)
        unlabel_pre_10_equal = np.r_[unlabel_pre_10_equal, i_unlabel_10_equal]
        unlabelmean.append(np.mean(i_unlabel_10_equal))
        unlabelstd.append(np.std(i_unlabel_10_equal))
    # print(np.shape(ratio_fn))
    # print(np.shape(label_pre_10_equal))
    # print(np.shape(labelmean))
    # print(np.shape(round5_ratio_unlabel_positive))
    # print(np.shape(labelmean))
    # print(np.shape(labelmean))
    # print(np.shape(labelmean))
    model_infor = np.hstack((ratio_tp, ratio_fp, ratio_tn, ratio_fn, label_pre_10_equal, labelmean, labelstd, \
         round5_ratio_unlabel_positive, round5_ratio_unlabel_negative, unlabel_pre_10_equal, unlabelmean, unlabelstd))
    print('model_infor', np.shape(model_infor))
    model_infor_data = model_infor * np.ones_like(n_feature_data)

    fx_data = None
    k = 0
    for i in query_index:
        f_x_a = []
        # f_x_b = []
        f_x_c = []
        f_x_d = []
        for round in range(6):
            predict = minmax_scale(modelPredictions[round])
            # for j in i_lcc_sort_index:
            #     f_x_a.append(predict[i] - predict[j])
            # for j in i_ucc_sort_index:
            #     f_x_b.append(predict[i] - predict[j])

            # for j in data_cluster_centers_10_index:
            #     f_x_a.append(predict[i] - predict[data_cluster_centers_10_index[cc_sort_index[i]]])
            # for j in range(10):
            #     f_x_a.append(predict[i] - predict[cc_sort_index[i][j]])

            for j in range(10):
                print('cc_sort_index[i][j]', cc_sort_index[k][j])
                f_x_a.append(predict[i] - currentmodel.predict(data_cluster_centers_10[cc_sort_index[k][j]]))
            for j in range(10):
                f_x_c.append(predict[i] - predict[label_10_equal_index[j]])
            for j in range(10):
                f_x_d.append(predict[i] - predict[unlabel_10_equal_index[j]])
        # fdata = np.hstack((f_x_a, f_x_b, f_x_c, f_x_d))
        fdata = np.hstack((current_prediction[i], f_x_a, f_x_c, f_x_d))
        if fx_data is None:
            fx_data = fdata
        else:
            fx_data = np.vstack((fx_data, fdata))
        k += 1
    # print('fx_data', np.shape(fx_data))
    # fx_data = fx_data * np.ones_like(n_feature_data)


    print(np.shape(n_feature_data))
    print(np.shape(ratio_label_positive_data))
    print(np.shape(ratio_label_negative_data))
    print(np.shape(ratio_unlabel_positive_data))
    print(np.shape(ratio_unlabel_negative_data))
    print(np.shape(distance_query_data))
    print(np.shape(model_infor_data))
    print(np.shape(fx_data))

    metadata = np.hstack((n_feature_data, ratio_label_positive_data, ratio_label_negative_data, \
         ratio_unlabel_positive_data, ratio_unlabel_negative_data, distance_query_data, model_infor_data, fx_data))
    print(np.shape(metadata))
    return metadata


if __name__ == "__main__":
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2)
    y[y==0] = -1
    labels = []
    for j in range(10,16):
        labels.append(np.array([i for i in range(j)]))

    unlabels = []
    for j in range(10,16):
        unlabels.append(np.array([i for i in range(j,100)]))


    # labels = np.array([i for i in range(10)])

    # labels = np.vstack((labels, np.array([i for i in range(11)])))
    # labels = np.vstack((labels, np.array([i for i in range(12)])))
    # labels = np.vstack((labels, np.array([i for i in range(13)])))
    # labels = np.vstack((labels, np.array([i for i in range(14)])))
    # labels = np.vstack((labels, np.array([i for i in range(15)])))

    # unlabels = np.array([i for i in range(10,100)])
    # unlabels = np.vstack((unlabels, np.array([i for i in range(11,100)])))
    # unlabels = np.vstack((unlabels, np.array([i for i in range(12,100)])))
    # unlabels = np.vstack((unlabels, np.array([i for i in range(13,100)])))
    # unlabels = np.vstack((unlabels, np.array([i for i in range(14,100)])))
    # unlabels = np.vstack((unlabels, np.array([i for i in range(15,100)])))
    
    models = []
    decision_value = []
    prediction = []
    label_ys = []
    # model = SVC()
    # model.fit(X[labels[0]], y[labels[0]])
    # pre = model.predict(X)
    # de = model.decision_function(X)
    # print(pre[10:20])
    # print(de[10:20])

    for i in range(6):
        model = SVC()
        model.fit(X[labels[i]], y[labels[i]])
        prediction.append(model.predict(X))
        # prediction = np.vstack(())
        decision_value.append(model.decision_function(X))
        # decision_value = np.vstack((decision_value, model.decision_function(X)))
        # label_ys = np.vstack((label_ys, y[labels[i]]))
        label_ys.append(y[labels[i]])
        models.append(model)
    
    query_index = [i for i in range(15, 21)]
    query_index = np.array(query_index)
    meta = matedata(X, label_ys, labels, unlabels, prediction, query_index, models[5])
