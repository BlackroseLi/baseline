import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer,minmax_scale

from sklearn.datasets import make_classification
from sklearn.svm import SVC

class DataSet():
    """

    Parameters
    ----------
    X: 2D
        The dataset.
    
    """
    def __init__(self, X):
        if not isinstance(X, (list, np.ndarray)):
            raise ValueError("")
        self.X = X

        self.n_samples, self.n_features =  np.shape(X)
        self.distance = None
    
    def get_cluster_center(self, n_clusters=10, method='Euclidean'):
        """Use the Kmeans in sklearn to get the cluster centers.

        Parameters
        ----------
        n_clusters: int 
            The number of cluster centers.
        Returns
        -------
        data_cluster_centers: np.ndarray
            The samples in origin dataset X is the closest to the cluster_centers.
        index_cluster_centers: np.ndarray
            The index corresponding to the samples in origin data set.     
        """
        if self.distance is None:
            self.get_distance()
        data_cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(self.X)
        data_origin_cluster_centers = data_cluster.cluster_centers_
        closest_distance_data_cluster_centers = np.zeros(n_clusters) + np.infty
        index_cluster_centers = np.zeros(n_clusters, dtype=int) - 1

        # obtain the cluster centers index
        for i in range(self.n_samples):
            for j in range(n_clusters):
                if method == 'Euclidean':
                    distance = np.linalg.norm(X[i] - data_origin_cluster_centers[j])
                    if distance < closest_distance_data_cluster_centers[j]:
                        closest_distance_data_cluster_centers[j] = distance
                        index_cluster_centers[j] = i

        if(np.any(index_cluster_centers == -1)):
            raise IndexError("data_cluster_centers_index is wrong")

        return X[index_cluster_centers], index_cluster_centers


    def get_distance(self, method='Euclidean'):
        """

        Parameters
        ----------
        method: str
            The method calculate the distance.
        Returns
        -------
        distance_martix: 2D
            D[i][j] reprensts the distance between X[i] and X[j].
        """
        if self.n_samples == 1:
            raise ValueError("There is only one sample.")
        
        distance = np.zeros((self.n_samples, self.n_samples))
        for i in range(1, self.n_samples):
            for j in range(i+1, self.n_samples):
                if method == 'Euclidean':
                    distance[i][j] = np.linalg.norm(self.X[i] - self.X[j])
        
        self.distance = distance + distance.T
        return self.distance


def mate_data(X, label_ys, label_indexs, unlabel_indexs, modelPredictions, query_index):
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

    for i in range(6):
        assert(len(label_ys[i]) == len(label_indexs[i]))
        assert(np.shape(X)[0] == np.shape(modelPredictions[i])[0]) 
        if(not isinstance(label_indexs[i], np.ndarray)):
            label_indexs[i] = np.array(label_indexs[i])
        if(not isinstance(unlabel_indexs[i], np.ndarray)):
            unlabel_indexs[i] = np.array(unlabel_indexs[i])
    
    n_samples, n_feature = np.shape(X)
    query_index_size = len(query_index)
    n_feature_data = n_feature * np.ones((query_index_size, 1))
    current_label_size = len(label_indexs[5])
    current_label_y = label_ys[5]
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


    # the same dataset the same cluster centers
    data_cluster = KMeans(n_clusters=10, random_state=0).fit(X)
    data_origin_cluster_centers_10 = data_cluster.cluster_centers_
    closest_distance_data_cluster_centers_10 = np.zeros(10) + np.infty
    data_cluster_centers_10_index = np.zeros(10, dtype=int) - 1

    # obtain the cluster centers index
    for i in range(n_samples):
        for j in range(10):
            distance = np.linalg.norm(X[i] - data_origin_cluster_centers_10[j])
            if distance < closest_distance_data_cluster_centers_10[j]:
                closest_distance_data_cluster_centers_10[j] = distance
                data_cluster_centers_10_index[j] = i

    data_cluster_centers_10 = X[data_cluster_centers_10_index]
    if(np.any(data_cluster_centers_10_index == -1)):
        raise IndexError("data_cluster_centers_10_index is wrong")
    print('data_cluster_centers_10_index', data_cluster_centers_10_index)
    
    sorted_labelperdiction_index = np.argsort(current_prediction[label_indexs[5]])
    sorted_current_label_data = X[label_indexs[5][sorted_labelperdiction_index]]
    
    label_10_equal = [sorted_current_label_data[int(i * current_label_size)] for i in np.arange(0, 1, 0.1)]
    label_10_equal_index = [label_indexs[5][sorted_labelperdiction_index][int(i * current_label_size)] for i in np.arange(0, 1, 0.1)]

    sorted_unlabelperdiction_index = np.argsort(current_prediction[unlabel_indexs[5]])
    sorted_current_unlabel_data = X[unlabel_indexs[5][sorted_unlabelperdiction_index]]
    unlabel_10_equal = [sorted_current_unlabel_data[int(i * current_unlabel_size)] for i in np.arange(0, 1, 0.1)]
    unlabel_10_equal_index = [unlabel_indexs[5][sorted_unlabelperdiction_index][int(i * current_unlabel_size)] for i in np.arange(0, 1, 0.1)]

              
    distance_query_data = None
    cc_sort_index = []

    for i in query_index:
        i_cc = []
        i_l10e = []
        i_u10e = []
        for j in range(10):
            # cal the ith in query_index about 
            i_cc.append(np.linalg.norm(X[i] - data_cluster_centers_10[j]))
            i_l10e.append(np.linalg.norm(X[i] - label_10_equal[j]))
            i_u10e.append(np.linalg.norm(X[i] - unlabel_10_equal[j]))

        i_cc = minmax_scale(i_cc)
        i_cc_sort_index = np.argsort(i_cc)
        cc_sort_index.append(i_cc_sort_index)
        i_l10e = minmax_scale(i_l10e)
        i_u10e = minmax_scale(i_u10e)
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
        labelmean.append(np.mean(i_label_10_equal))
        labelstd.append(np.std(i_label_10_equal))

        round5_ratio_unlabel_positive.append((sum(current_prediction[unlabel_ind] > 0)) / unlabel_size)
        round5_ratio_unlabel_negative.append((sum(current_prediction[unlabel_ind] < 0)) / unlabel_size)
        sort_unlabel_pred = np.sort(minmax_scale(cur_prediction[unlabel_ind]))
        i_unlabel_10_equal = [sort_unlabel_pred[int(i * unlabel_size)] for i in np.arange(0, 1, 0.1)]
        unlabel_pre_10_equal = np.r_[unlabel_pre_10_equal, i_unlabel_10_equal]
        unlabelmean.append(np.mean(i_unlabel_10_equal))
        unlabelstd.append(np.std(i_unlabel_10_equal))
    model_infor = np.hstack((ratio_tp, ratio_fp, ratio_tn, ratio_fn, label_pre_10_equal, labelmean, labelstd, \
         round5_ratio_unlabel_positive, round5_ratio_unlabel_negative, unlabel_pre_10_equal, unlabelmean, unlabelstd))
    model_infor_data = model_infor * np.ones_like(n_feature_data)

    fx_data = None
    k = 0
    for i in query_index:
        f_x_a = []
        # f_x_b = []
        f_x_c = []
        f_x_d = []
        # print('data_cluster_centers_10_index[cc_sort_index[k]]', data_cluster_centers_10_index[cc_sort_index[k]])
        for round in range(6):
            predict = minmax_scale(modelPredictions[round])
            for j in range(10):
                f_x_a.append(predict[i] - predict[data_cluster_centers_10_index[cc_sort_index[k][j]]])
            for j in range(10):
                f_x_c.append(predict[i] - predict[label_10_equal_index[j]])
            for j in range(10):
                f_x_d.append(predict[i] - predict[unlabel_10_equal_index[j]])
        fdata = np.hstack((current_prediction[i], f_x_a, f_x_c, f_x_d))
        if fx_data is None:
            fx_data = fdata
        else:
            fx_data = np.vstack((fx_data, fdata))
        k += 1

    metadata = np.hstack((n_feature_data, ratio_label_positive_data, ratio_label_negative_data, \
         ratio_unlabel_positive_data, ratio_unlabel_negative_data, distance_query_data, model_infor_data, fx_data))
    print('The shape of meta_data: ', np.shape(metadata))
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
    
    models = []
    decision_value = []
    prediction = []
    label_ys = []

    for i in range(6):
        model = SVC()
        model.fit(X[labels[i]], y[labels[i]])
        prediction.append(model.predict(X))
        decision_value.append(model.decision_function(X))
        label_ys.append(y[labels[i]])
        models.append(model)
    
    query_index = [i for i in range(15, 21)]
    query_index = np.array(query_index)
    meta = mate_data(X, label_ys, labels, unlabels, prediction, query_index)
    d = DataSet(X)
    cd, cdi = d.get_cluster_center()
    print('cdi', cdi)
    

