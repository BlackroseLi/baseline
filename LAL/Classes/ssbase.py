import os
os.chdir(r'C:\Users\31236\Desktop\baseline\LAL')

import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from sgd_linear import SGDLinear
from sklearn.utils import resample

class SemiSupervisedBase:

    def __init__(self, filename, data, method = "random"):
        # Configuration variables.
        self.num_runs = 10 # Number of runs to average for results.
        self.num_committee = 4 # Size of the committee for QBC.
        self.num_iterations = 3 # Number of active learning loops.
        self.label_percent = 0.1 # Percent of labeled data.
        self.test_percent = 0.2 # Percent of test data.
        self.batch_percent = 0.03 # Percent of data to add to labeled data in each loop.
        # Initialize variables.
        self.cache = None # Used to cache values to speed up iterations.
        self.method = method # Name of active learning method.
        # Read data.
        # with open("data/{}.dat".format(name), "rb") as infile:
        #     self.data = pickle.loads(infile.read())
        # self.data = None
        self.data = data
        self.get_data(filename, batch_percent=0.2, label_percent=0.2, test_percent=0.2)


    def get_data(self, filename, batch_percent=0.2, label_percent=0.2, test_percent=0.2):

        data = np.load(filename)

        regression_features = data['arr_0']
        regression_labels = np.transpose([data['arr_1']])

        self.data['data'] = regression_features
        self.data['target'] = regression_labels
        
        # Reset cache values
        self.cache = None
        # Get counts for different sets.
        count = regression_features.shape[0]
        labeled_count = int(count * label_percent)
        test_count = int(count * test_percent)
        unlabeled_count = count - labeled_count - test_count
        self.batch_count = int(count * batch_percent)
        pos_list = list(range(count))
        # Split the data into training/testing sets
        random.shuffle(pos_list)
        self.labeled_pos_list = pos_list[:labeled_count]
        self.unlabeled_pos_list = pos_list[labeled_count:(labeled_count+unlabeled_count)]
        self.test_pos_list = pos_list[(labeled_count+unlabeled_count):]

    def sequential_select(self):
        """
        return the select index
        """
        # Build the committee.
        models = []
        for i in range(self.num_committee):
            # Build bootstrap of training data.
            bootstrap_labeled_pos_list = resample(self.labeled_pos_list, random_state=random.randrange(1000000))

            data_X_train = self.data["data"][ bootstrap_labeled_pos_list ]

            # Split the targets into training/testing sets
            data_y_train = self.data["target"][ bootstrap_labeled_pos_list ]

            # Create linear regression object
            model = SGDLinear()

            # Train the model using the training sets
            model.fit(data_X_train, data_y_train)
            models.append(model)

        y_act = {}
        y_est = {}
        eq_24 = {}
        for pos in self.unlabeled_pos_list:
            x = self.data["data"][ [pos] , : ]
            fx = self.model.predict(x)
            y_act[pos] = self.data["target"][pos]
            y_est[pos] = fx
            eq_24[pos] = 0
            for j in range(len(models)):
                y = models[j].predict(x)
                eq_24[pos] += np.linalg.norm((fx - y) * x)
            eq_24[pos] /= (1.0 * len(models))

        max_change = -1
        max_pos = None
        for pos in self.unlabeled_pos_list:
            change = eq_24[pos]
            if change > max_change:
                max_pos = pos
                max_change = change
        self.labeled_pos_list.append(max_pos)
        self.unlabeled_pos_list.remove(max_pos)  

        return max_pos

    def select(self):

        # Use linear regression using SGD
        self.model = SGDLinear()
        beforeselect_rmse = self.train()
        selected_index = []
        for i in range(self.batch_count):
            s = self.sequential_select()
            selected_index.append(s)

        data_X_test = self.data["data"][ self.test_pos_list ]
        data_y_test = self.data["target"][ self.test_pos_list ]
        data_y_pred = self.model.predict(X = data_X_test)
        after_rmse = get_root_mean_squared(data_y_test, data_y_pred)
        return beforeselect_rmse, after_rmse, selected_index

    def get_average(self):
        print("Start process for {} {}...".format(self.name, self.method))
        rmse = []
        for i in range(self.num_runs):
            random.seed(i * 473)
            np.random.seed(i * 473)
            rmse.append(self.process())
        rmse = np.array(rmse)

        # Calculate Average
        N = rmse.shape[0]
        M = rmse.shape[1]
        x = list(range(M))
        y_average = np.zeros(M)
        for i in range(N):
            y_average += rmse[i]
        y_average /= N

        # Calculate Standard Deviation
        y_stddev = np.zeros(M)
        for i in range(N):
            y_stddev += (rmse[i] - y_average) * (rmse[i] - y_average)
        y_stddev = np.sqrt(y_stddev / N)

        # Write output.
        with open("results/{}.txt".format(self.name + "_" + self.method), "w") as outfile:
            outfile.write("iteration\t{}\n".format(self.method))
            for i in range(rmse.shape[0]):
                for j in range(rmse.shape[1]):
                    outfile.write(str(j) + "\t" + str(rmse[i, j]) + "\n")

        # Build 1 stddev.
        y_top = y_average + y_stddev
        y_bottom = y_average - y_stddev

        # Plot range
        fig, ax = plt.subplots()
        ax.plot(x, y_average, color="black")
        ax.plot(x, y_top, x, y_bottom, color="black")
        ax.fill_between(x, y_average, y_top, where=y_top>y_average, facecolor="green", alpha=0.5)
        ax.fill_between(x, y_average, y_bottom, where=y_bottom<=y_average, facecolor="red", alpha=0.5)
        plt.show()

    def train(self):
        data_X_train = self.data["data"][ self.labeled_pos_list ]
        data_X_test = self.data["data"][ self.test_pos_list ]

        # Split the targets into training/testing sets
        data_y_train = self.data["target"][ self.labeled_pos_list ]
        data_y_test = self.data["target"][ self.test_pos_list ]

        # Train the model using the training sets
        self.model.fit(train_x = data_X_train, train_y = data_y_train)

        # Make predictions using the testing set
        data_y_pred = self.model.predict(X = data_X_test)
        #data_y_pred = self.model.predict(X = data_X_train)

        # Get prediction error using mean absolute error.
        rmse = get_root_mean_squared(data_y_test, data_y_pred)
        #rmse = get_root_mean_squared(data_y_train, data_y_pred)
        return rmse

    def update_labeled(self):
        if self.method == "random":
            self.update_labeled_random()
        # elif self.method == "greedy":
        #     self.update_labeled_greedy()
        # elif self.method == "greedy2":
        #     self.update_labeled_greedy2()
        elif self.method == "qbc":
            self.update_labeled_qbc()
        elif self.method == "qbc2":
            self.update_labeled_qbc2()
        elif self.method == "bemcm":
            return self.update_labeled_bemcm()
        else:
            print("Method '{}' is unknown.".format(self.method))
            exit()

    def update_labeled_random(self):
        self.labeled_pos_list.extend(self.unlabeled_pos_list[:self.batch_count])
        self.unlabeled_pos_list = self.unlabeled_pos_list[self.batch_count:]
        #print("Random Update {:.2f}s".format(total_time))

    
   

    def update_labeled_bemcm(self):
        """
        return the select index
        """
        # Build the committee.
        models = []
        for i in range(self.num_committee):
            # Build bootstrap of training data.
            bootstrap_labeled_pos_list = resample(self.labeled_pos_list, random_state=random.randrange(1000000))

            data_X_train = self.data["data"][ bootstrap_labeled_pos_list ]

            # Split the targets into training/testing sets
            data_y_train = self.data["target"][ bootstrap_labeled_pos_list ]

            # Create linear regression object
            model = SGDLinear()

            # Train the model using the training sets
            model.fit(data_X_train, data_y_train)
            models.append(model)

        y_act = {}
        y_est = {}
        eq_24 = {}
        for pos in self.unlabeled_pos_list:
            x = self.data["data"][ [pos] , : ]
            fx = self.model.predict(x)
            y_act[pos] = self.data["target"][pos]
            y_est[pos] = fx
            eq_24[pos] = 0
            for j in range(len(models)):
                y = models[j].predict(x)
                eq_24[pos] += np.linalg.norm((fx - y) * x)
            eq_24[pos] /= (1.0 * len(models))

        select_index = []
        for i in range(self.batch_count):
            max_change = -1
            max_pos = None
            for pos in self.unlabeled_pos_list:
                change = eq_24[pos]
                if change > max_change:
                    max_pos = pos
                    max_change = change
            select_index.append(max_pos)
            del eq_24[max_pos]

            self.labeled_pos_list.append(max_pos)
            self.unlabeled_pos_list.remove(max_pos)
        return select_index

    def update_labeled_qbc(self):
        # Build the committee.
        models = []
        for i in range(self.num_committee):
            # Build bootstrap of training data.
            bootstrap_labeled_pos_list = resample(self.labeled_pos_list, random_state=random.randrange(1000000))
            # Get bootstrap training set.
            data_X_train = self.data["data"][ bootstrap_labeled_pos_list ]
            # Get bootstrap target set.
            data_y_train = self.data["target"][ bootstrap_labeled_pos_list ]
            # Create linear regression object
            model = SGDLinear()
            # Train the model using the training sets
            model.fit(data_X_train, data_y_train)
            models.append(model)

        variances = []
        for pos in self.unlabeled_pos_list:
            variance = 0
            y_ave = 0
            for model in models:
                y = model.predict(self.data["data"][ [pos], :])
                variance += y * y
                y_ave += y
            y_ave /= (len(models) * 1.0)
            variance /= (len(models) * 1.0)
            variance -= y_ave * y_ave
            variances.append((variance, pos))

        variances.sort(reverse = True)
        for i in range(self.batch_count):
            self.labeled_pos_list.append(variances[i][1])
            self.unlabeled_pos_list.remove(variances[i][1])
        x = []
        x.extend(self.labeled_pos_list)
        x.extend(self.unlabeled_pos_list)
        x.extend(self.test_pos_list)
        #print("QBC Update {:.2f}s".format(total_time))

    def update_labeled_qbc2(self):
        # Build the committee.
        for i in range(self.batch_count):
            models = []
            for i in range(self.num_committee):
                # Build bootstrap of training data.
                bootstrap_labeled_pos_list = resample(self.labeled_pos_list, random_state=random.randrange(1000000))
                # Get bootstrap training set.
                data_X_train = self.data["data"][ bootstrap_labeled_pos_list ]
                # Get bootstrap target set.
                data_y_train = self.data["target"][ bootstrap_labeled_pos_list ]
                # Create linear regression object
                model = SGDLinear()
                # Train the model using the training sets
                model.fit(data_X_train, data_y_train)
                models.append(model)

            max_variance = 0
            max_pos = -1
            for pos in self.unlabeled_pos_list:
                variance = 0
                y_ave = 0
                for model in models:
                    y = model.predict(self.data["data"][ [pos], :])
                    variance += y * y
                    y_ave += y
                y_ave /= (len(models) * 1.0)
                variance /= (len(models) * 1.0)
                variance -= y_ave * y_ave
                if variance > max_variance:
                    max_variance = variance
                    max_pos = pos
            self.labeled_pos_list.append(max_pos)
            self.unlabeled_pos_list.remove(max_pos)
        #print("QBC2 Update {:.2f}s".format(total_time))

    def get_min_distance(self, i):
        min_dist = None
        min_pos = -1
        for j in self.labeled_pos_list:
            dist = self.calc_distance(i, j)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_pos = j
        return min_dist

    def calc_distance(self, i, j):
        key = (i, j)
        if self.cache is None:
            self.cache = {}
        if key not in self.cache:
            x = self.data["data"][i]
            y = self.data["data"][j]
            self.cache[key] = np.linalg.norm(x-y)
        x = self.cache[key]
        return x


def get_mean_absolute_error(y_actual, y_predict):
    T = y_actual.shape[0]
    mae = np.sum(abs(y_actual - y_predict)) / T
    return mae


def get_root_mean_squared(y_actual, y_predict):
    T = y_actual.shape[0]
    rmse = math.sqrt(np.transpose(y_actual - y_predict) * (y_actual - y_predict) / T)
    return rmse

if __name__ == "__main__":
    data = {'data':np.array([]),'traget':np.array([])}
    s = SemiSupervisedBase('./lal datasets/LAL-iterativetree-simulatedunbalanced-big.npz', data, "bemcm")
    befor_rmse, after_rmse, select_results = s.select()

    print(befor_rmse)
    print(after_rmse)
    print(np.shape(select_results))
    print(select_results[0])
    
    # print(rmse_list[0:3])
    # print(select_results[0])

    # filename = './lal datasets/LAL-iterativetree-simulatedunbalanced-big.npz'
    # batch_percent=0.2
    # label_percent=0.2 
    # test_percent=0.2

    # data1 = np.load(filename)

    # regression_features = data1['arr_0']
    # regression_labels = np.transpose([data1['arr_1']])

    # print(type(regression_features))
    # print(regression_features[0])

    # data['data'] = regression_features
    # data['target'] = regression_labels
    # print(np.shape(data['data']))
    # print(np.shape(data['target']))
    # # Get counts for different sets.
    # count = regression_features.shape[0]
    # labeled_count = int(count * label_percent)
    # test_count = int(count * test_percent)
    # unlabeled_count = count - labeled_count - test_count
    # batch_count = int(count * batch_percent)
    # pos_list = list(range(count))
    # # Split the data into training/testing sets
    # random.shuffle(pos_list)
    # labeled_pos_list = pos_list[:labeled_count]
    # unlabeled_pos_list = pos_list[labeled_count:(labeled_count+unlabeled_count)]
    # test_pos_list = pos_list[(labeled_count+unlabeled_count):]


    # print(data['data'][labeled_pos_list].shape)
    # print(data['target'][labeled_pos_list].shape)
    # print(data['data'][unlabeled_pos_list].shape)
    # print(data['target'][unlabeled_pos_list].shape)
    # print(data['data'][test_pos_list].shape)
    # print(data['target'][test_pos_list].shape)
