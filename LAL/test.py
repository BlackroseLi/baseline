import os
os.chdir(r'C:\Users\31236\Desktop\baseline\LAL')

import numpy as np 

fn = 'LAL-randomtree-simulatedunbalanced-big.npz'
# we found these parameters by cross-validating the regressor and now we reuse these expreiments
parameters = {'est': 2000, 'depth': 40, 'feat': 6 }
filename = './lal datasets/'+fn
regression_data = np.load(filename)
regression_features = regression_data['arr_0']
regression_labels = regression_data['arr_1']

print(np.shape(regression_features))
print(regression_features[0,:])
print(np.shape(regression_labels))
print(regression_labels[0:5])
