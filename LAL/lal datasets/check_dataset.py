import numpy as np 

fn = 'LAL-randomtree-simulatedunbalanced-big.npz'
filename = './' + fn
filename = './lal datasets/'+fn
lal_iterativetree = np.load(fn)

regression_features = lal_iterativetree['arr_0']
regression_labels = lal_iterativetree['arr_1']

# print('type of :', type(lal_iterativetree))
# print('the shape of lal_iterativetree', lal_iterativetree)

print('shape of regression_features    ', regression_features.shape)
print('regression_features[0]', regression_features[0])
print('shape of regression_labels   ', regression_labels.shape)
print('regression_labels[0: 10]', regression_labels[0: 10])