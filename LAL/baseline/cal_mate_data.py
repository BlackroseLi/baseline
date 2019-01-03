import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer,minmax_scale
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





def matedata(X, label_ys, label_indexs, unlabel_indexs, modelPredictions, query_index):
    """


    """
    n_samples, n_features = np.shape(X)

    current_label_data = X[label_indexs[5]]
    current_label_size = len(label_indexs[5])
    current_label_y = label_ys[5]
    current_unlabel_data = X[unlabel_indexs[5]]
    current_unlabel_size = len(unlabel_indexs[5])
    current_prediction = modelPredictions[5]

    ratio_label_positive = (sum(current_label_y > 0)) / current_label_size
    ratio_label_negative = (sum(current_label_y < 0)) / current_label_size

    ratio_unlabel_positive = (sum(current_prediction[unlabel_indexs[5]] > 0)) / current_unlabel_size
    ratio_unlabel_negative = (sum(current_prediction[unlabel_indexs[5]] < 0)) / current_unlabel_size

    label_cluster = KMeans(n_clusters=10).fit(current_label_data)
    label_cluster_centers_10 = label_cluster.cluster_centers_
    label_cluster_centers_10_index = 
    unlabel_cluster = KMeans(n_clusters=10).fit(current_unlabel_data)
    unlabel_cluster_centers_10 = unlabel_cluster.cluster_centers_
    unlabel_cluster_centers_10_index = 
    
    sorted_current_label_data = np.sort(current_label_data)
    label_10_equal = [ sorted_current_label_data[int(round(i * current_label_size))] for i in np.arange(0, 1, 0.1)]
    sorted_current_unlabel_data = np.sort(current_label_data)
    unlabel_10_equal = [ sorted_current_unlabel_data[int(round(i * current_unlabel_size))] for i in np.arange(0, 1, 0.1)]

    distance_query_index = np.array()
    for i in query_index:
        i_lcc = []
        i_ucc = []
        i_l10e = []
        i_u10e = []

        # f_i_lcc = []
        # f_i_ucc = []
        # f_i_l10e = []
        # f_i_u10e = []      
        for j in range(10):
            # cal the ith in query_index about 
            i_lcc.append(np.linalg.norm(current_unlabel_data[i] - label_cluster_centers_10[j]))
            i_ucc.append(np.linalg.norm(current_unlabel_data[i] - unlabel_cluster_centers_10[j]))
            i_l10e.append(np.linalg.norm(current_unlabel_data[i] - label_10_equal[j]))
            i_u10e.append(np.linalg.norm(current_unlabel_data[i] - unlabel_10_equal[j]))
        
        i_lcc = minmax_scale(i_lcc)
        i_ucc = minmax_scale(i_ucc)
        i_l10e = minmax_scale(i_l10e)
        i_u10e = minmax_scale(i_u10e)
        i_distance = np.hstack((i_lcc, i_ucc, i_l10e, i_u10e))
        np.vstack((distance_query_index, i_distance))

        
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
    for i in range(5):
        label_size = len(label_indexs[i])
        unlabel_size = len(unlabel_indexs[i])
        tn, fp, fn, tp = confusion_matrix(label_ys[i], modelPredictions[i][label_indexs[i]], labels=[-1, 1]).ravel()
        ratio_tn.append(tn / label_size)
        ratio_fp.append(fp / label_size)
        ratio_fn.append(fn / label_size)
        ratio_tp.append(tp / label_size)
        cur_prediction = modelPredictions[i]
        label_ind = label_indexs[i]
        
        unlabel_ind = unlabel_indexs[i]

        sort_label_pred = np.sort(minmax_scale(cur_prediction[label_ind]))
        i_label_10_equal = [ sort_label_pred[int(round(i * label_size))] for i in np.arange(0, 1, 0.1)]
        label_pre_10_equal.append(i_label_10_equal)
        labelmean.append(np.mean(i_label_10_equal))
        labelstd.append(np.std(i_label_10_equal))

        round5_ratio_unlabel_positive.append((sum(current_prediction[unlabel_indexs[i]] > 0)) / unlabel_size)
        round5_ratio_unlabel_negative.append((sum(current_prediction[unlabel_indexs[i]] < 0)) / unlabel_size)
        sort_unlabel_pred = np.sort(minmax_scale(cur_prediction[unlabel_ind]))
        i_unlabel_10_equal = [ sort_unlabel_pred[int(round(i * unlabel_size))] for i in np.arange(0, 1, 0.1)]
        unlabel_pre_10_equal.append(i_unlabel_10_equal)
        unlabelmean.append(np.mean(i_unlabel_10_equal))
        unlabelstd.append(np.std(i_unlabel_10_equal))


