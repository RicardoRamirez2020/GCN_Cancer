import sys, os
sys.path.insert(0, '..')
from lib import models, graph, coarsening, utils
import sklearn.metrics
import sklearn.neighbors
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import tensorflow as tf
import numpy as np
import time
import h5py
import scipy.io as sio

#matplotlib inline

flags = tf.app.flags
FLAGS = flags.FLAGS

# Graphs.
flags.DEFINE_integer('number_edges', 1, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 1, 'Number of coarsened graphs.')



t_start = time.process_time()
# Directories.
flags.DEFINE_string('dir_data', os.path.join('..', 'data', 'mnist'), 'Directory to store data.')
#For PPI and PPI-singleton model change file location
test = sio.loadmat('C:/Users/RJ\Desktop/exp_fpkm_pancan/processed/PPI_filtered/Adj_Filtered_List_0Con.mat')
# for Correlaton model change file location
#test = sio.loadmat('C:/Users/RJ\Desktop/exp_fpkm_pancan/processed/CoExpression/Adj_Data/Adj_Spearman_6P.mat')
row = test['row'].astype(np.float32)
col = test['col'].astype(np.float32)
value = test['value'].astype(np.float32)
M, k = row.shape
row = np.array(row)
row = row.reshape(k)
row = row.ravel()
col = np.array(col)
col = col.reshape(k)
col = col.ravel()
value = np.array(value)
value = value.reshape(k)
value = value.ravel()
A = scipy.sparse.coo_matrix((value, (row, col)),shape = (4444,4444)) # change size for model being used 4444 for both PPI and 3866 for
graphs, perm = coarsening.coarsen(A, levels=FLAGS.coarsening_levels, self_connections=True)
L = [graph.laplacian(A, normalized=True,renormalized=True) for A in graphs]
del test
del A
del row
del col
del value

Data = sio.loadmat('C:/Users/RJ/Desktop/exp_fpkm_pancan/processed/Final/Data/Block_PPIA.mat')
Data1 = Data['Block'][0,0]
Data2 = Data['Block'][0,1]
Data3 = Data['Block'][0,2]
Data4 = Data['Block'][0,3]
Data5 = Data['Block'][0,4]
D1= Data1['D'].astype(np.float32)
D2= Data2['D'].astype(np.float32)
D3= Data3['D'].astype(np.float32)
D4= Data4['D'].astype(np.float32)
D5= Data5['D'].astype(np.float32)
L1= Data1['L'].astype(np.float32)
L2= Data2['L'].astype(np.float32)
L3= Data3['L'].astype(np.float32)
L4= Data4['L'].astype(np.float32)
L5= Data5['L'].astype(np.float32)
# adjust for K-Fold cross validation
Train_Data = np.transpose(np.hstack((D1,D2,D3,D4)))
Val_Data = np.transpose(D5)
Test_Data = np.transpose(D5)
Train_Label = (np.vstack((L1,L2,L3,L4)))
Val_Label = (L5)
Test_Label = (L5)
Test_Label = Test_Label.ravel()
Train_Label = Train_Label.ravel()
Val_Label = Val_Label.ravel()

Train_Data = coarsening.perm_data(Train_Data, perm)
Val_Data = coarsening.perm_data(Val_Data, perm)
Test_Data = coarsening.perm_data(Test_Data, perm)


C = 34  # number of classes

common = {}
common['dir_name']       = 'PPI/'
common['num_epochs']     = 20
common['batch_size']     = 200
common['decay_steps']    = 17.7 # * common['num_epochs'] since not used use as in momentum 
common['eval_frequency'] = 10 * common['num_epochs']
common['brelu']          = 'b1relu'
common['pool']           = 'apool1'

model_perf = utils.model_perf()

common['regularization'] = 0
common['dropout']        = 1
common['learning_rate']  = .005
common['decay_rate']     = 0.95
common['momentum']       = 0

common['F']              = [1]
common['K']              = [1]
common['p']              = [2]
common['M']              = [1024,C]


if True:
    name = 'Run1'
    params = common.copy()
    params['dir_name'] += name
#    params['filter'] = 'chebyshev5'
    params['filter'] = 'chebyshev2'
    params['brelu'] = 'b1relu'
    model_perf.test(models.cgcnn(L, **params), name, params, Train_Data, Train_Label, Val_Data, Val_Label, Test_Data, Test_Label)

model_perf.show()

if False:
    grid_params = {}
    data = (train_data, train_labels, val_data, val_labels, test_data, test_labels)
    utils.grid_search(params, grid_params, *data, model=lambda x: models.cgcnn(L,**x))
    