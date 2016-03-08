
# coding: utf-8

# In[1]:

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
# display plots in this notebook
get_ipython().magic(u'matplotlib inline')

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

model_name = "alex_dm-fc_8"
model_path = caffe_root + "models/deep_mammo/alex_train_val.prototxt"
pretrained_weights_path = caffe_root + "models/deep_mammo/bvlc_alexnet.caffemodel"
quick_solver_path = caffe_root + "models/deep_mammo/alex_solver.prototxt"
weight_dir = caffe_root + "models/deep_mammo/weights"
pickle_dir = caffe_root + "models/deep_mammo/pickles"
figures_dir = caffe_root + "models/deep_mammo/figures"
# net = caffe.Net(model_path, pretrained_weights_path, caffe.TRAIN)


# In[4]:

def run_solvers(niter, solvers, disp_interval=50):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    train_loss, train_acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    val_loss, val_acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    print "#,Solver,TrainBatchLoss,TrainBatchAccuracy,ValLoss,ValAccuracy"
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
    # Save the learned weights from all nets.
    for name, s in solvers:
        uid = "%s.%d_iter.%s" % (name, niter, model_name)
        weights_filename = '%s.caffemodel' % uid
        s.net.save(os.path.join(weight_dir, weights_filename))
        pickle_filename = '%s.pickle' % uid
        with open(os.path.join(pickle_dir, pickle_filename), 'w+') as f:
            pickle.dump([train_loss, train_acc, val_loss, val_acc], f)
    return train_loss, train_acc, val_loss, val_acc


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


# In[6]:

niter = 2275 # 200 epochs  # number of iterations to train

quick_solver = caffe.get_solver(quick_solver_path)
quick_solver.net.copy_from(pretrained_weights_path)

print 'Running solvers for %d iterations...' % niter
solvers = [('pretrained', quick_solver)]
train_loss, train_acc, val_loss, val_acc = run_solvers(niter, solvers)
print 'Done.'

pre_train_loss = train_loss['pretrained']
pre_train_acc = train_acc['pretrained']
pre_val_loss = val_loss['pretrained']
pre_val_acc = val_acc['pretrained']

# Delete solvers to save memory.
del quick_solver, solvers


# In[7]:

plt.plot(pre_train_loss.T)
plt.plot(pre_val_loss.T)
plt.xlabel('Iteration #')
plt.ylabel('Loss')
plt.legend(['Train Loss', 'Val Loss'])
plt.grid()
plt.savefig('%s/loss_curves_%d_iter_%s.png' %(figures_dir, niter, model_name))


# In[8]:

plt.plot(pre_train_acc.T)
plt.plot(pre_val_acc.T)
plt.xlabel('Iteration #')
plt.ylabel('Accuracy')
plt.legend(['Train Accuracy', 'Val Accuracy'])
plt.grid()
plt.savefig('%s/accuracy_curves_%d_iter_%s.png' %(figures_dir, niter, model_name))

