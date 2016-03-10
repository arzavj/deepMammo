
# coding: utf-8

# In[1]:

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from optparse import OptionParser
import lmdb
from caffe.proto import caffe_pb2

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# In[2]:

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = ""
sys.path.insert(0, caffe_root + 'python')

import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.
import tempfile

caffe.set_device(0)
caffe.set_mode_gpu()


# In[3]:

weight_dir = "/mnt/weights/" # caffe_root + "models/deep_mammo/weights"
pickle_dir = caffe_root + "models/deep_mammo/pickles"
figures_dir = caffe_root + "models/deep_mammo/figures"
val_lmdb = caffe_root + "../2x_padding_224x224_dataset/mass_2x_padding_dataset_val_lmdb"
# net = caffe.Net(model_path, pretrained_weights_path, caffe.TRAIN)


def save_snapshot(solvers, model_name, niter):
    weights = {}
    for name, s in solvers:
        uid = "%s.%d_iter.%s" % (name, niter, model_name)
        weights[name] = os.path.join(weight_dir, '%s.caffemodel' % uid)
        s.net.save(weights[name])
    return weights

def run_solvers(niter, solvers, model_name, disp_interval=50):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    train_loss, train_acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    val_loss, val_acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    snapshot_iter = int(niter / 4)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            train_loss[name][it], train_acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
            val_loss[name][it], val_acc[name][it] = eval_val_accuracy(s.test_nets[0])
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = ','.join('%s,%.3f,%2d%%,%.3f,%2d%%' %
                                  (n, train_loss[n][it], np.round(100*train_acc[n][it]), \
                                    val_loss[n][it], np.round(100*val_acc[n][it]))
                                  for n, _ in solvers)
            print '%3d,%s' % (it, loss_disp)
        if (it % snapshot_iter) == 0 and it != 0:
            save_snapshot(solvers, model_name, it)
    # Save the learned weights from all nets.
    weights = save_snapshot(solvers, model_name, niter)
    return train_loss, train_acc, val_loss, val_acc, weights


# In[5]:

def eval_val_accuracy(test_net, test_iters=9):
    accuracy = 0.0
    loss = 0.0
    for it in xrange(test_iters):
        outputBlobs = test_net.forward()
        accuracy += outputBlobs['acc']
        loss += outputBlobs['loss']
    accuracy /= test_iters
    loss /= test_iters
    return loss, accuracy

def get_val_data():
    lmdb_env = lmdb.open(val_lmdb)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    lmdb_cursor.first()
    datum = caffe_pb2.Datum()
    y_true = []
    val_images = []

    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        y_true.append(datum.label)
        data = caffe.io.datum_to_array(datum)
        #CxHxW to HxWxC in cv2
        val_images.append(np.transpose(data, (1,2,0)))
    return val_images, y_true

def get_val_predictions(deploy_filename, weights_filename):
    val_images, y_true = get_val_data()
    classifier = caffe.Classifier(deploy_filename, weights_filename)
    y_pred = classifier.predict(val_images)
    return y_true, y_pred

def getOptions():
    '''
    Get command-line options and handle errors.
    :return: command-line options and arguments
    '''
    parser = OptionParser()
    parser.add_option('-i', '--iter', type='int', dest='niter',
                      default=2844, help='use a dataset of size NUM',
                      metavar='NUM')
    parser.add_option('-p', '--prototxt', dest='model_path', default=caffe_root + "models/deep_mammo/alex_train_val.prototxt",
                      help='path name to train_val prototoxt', metavar='MODEL')
    parser.add_option('-w', '--weights', dest='weights_path', default=caffe_root + "models/deep_mammo/bvlc_alexnet.caffemodel",
                      help='path name to weights caffemodel', metavar='WEIGHTS')
    parser.add_option('-s', '--solver', dest='solver_path', default=caffe_root + "models/deep_mammo/alex_solver.prototxt",
                      help='path name to solver prototoxt', metavar='SOLVER')
    parser.add_option('-n', '--name', dest='model_name', default="alex_dm-fc_8",
                      help='unique identifier for model', metavar='MODEL_NAME')
    parser.add_option('-d', '--deploy', dest='deploy', default=caffe_root + "models/deep_mammo/alex_deploy.prototxt",
                      help='path name to deploy prototoxt', metavar='DEPLOY')
    options, args = parser.parse_args()

    return options, args

def main():
    options, args = getOptions()

    # In[6]:

    niter = options.niter # 200 epochs  # number of iterations to train

    quick_solver = caffe.get_solver(options.solver_path)
    quick_solver.net.copy_from(options.weights_path)

    print "Options: %s" % options
    print 'Running solvers for %d iterations...' % niter
    solvers = [('pretrained', quick_solver)]
    train_loss, train_acc, val_loss, val_acc, weights = run_solvers(niter, solvers, options.model_name)
    
    
    y_true, y_pred = get_val_predictions(options.deploy, weights['pretrained'])
    pickle_filename = "%d_iter.%s.pickle" % (niter, options.model_name)
    with open(os.path.join(pickle_dir, pickle_filename), 'w+') as f:
        pickle.dump([train_loss, train_acc, val_loss, val_acc, y_true, y_pred], f)

if __name__ == '__main__':
    main()
