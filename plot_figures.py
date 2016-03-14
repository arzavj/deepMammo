import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from optparse import OptionParser

pickles_dir = "pickles"
figures_dir = "figures"

def get_samples(array, interval):
    iters = np.linspace(0, array.shape[0], num=array.shape[0]/interval, endpoint=False, dtype=np.uint32)
    ret = array[iters]
    return iters, ret

def plot_loss(loss, filename, options):
    iters, loss = get_samples(loss, int(options.interval))
    plt.plot(iters, loss)
    plt.xlabel('Iteration #')
    plt.ylabel('Training Loss')
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig('%s/loss_curves_%s.png' % (figures_dir, filename))
    plt.close()

def plot_accuracy(train_acc, val_acc, filename, options):
    iters, train_acc = get_samples(train_acc, int(options.interval))
    iters, val_acc = get_samples(val_acc, int(options.interval))
    plt.plot(iters, train_acc)
    plt.plot(iters, val_acc)
    plt.xlabel('Iteration #')
    plt.ylabel('Accuracy')
    plt.legend(['Train Accuracy', 'Val Accuracy'], loc=4)
    plt.grid()
    plt.savefig('%s/accuracy_curves_%s.png' % (figures_dir, filename))
    plt.close()

def plot_all_pickles_in_directory(options):
    for f in os.listdir(pickles_dir):
        filename = os.path.join(pickles_dir, f)
        if os.path.isfile(filename):
            plot_pickle(options)

def plot_pickle(options):
    filename = options.pickle_filename
    with open(filename, 'r') as f:
        train_loss, train_acc, val_loss, val_acc = pickle.load(f)
        filename_without_extension = os.path.splitext(filename)[0].split("/")[1]
        plot_loss(train_loss['pretrained'], filename_without_extension, options)
        plot_accuracy(train_acc['pretrained'], val_acc['pretrained'], filename_without_extension, options)

def getOptions():
    '''
    Get command-line options and handle errors.
    :return: command-line options and arguments
    '''
    parser = OptionParser()
    parser.add_option('-p', '--pickle', dest='pickle_filename',
                      help='path name to pickle file', metavar='PICKLE')
    parser.add_option('-a', action="store_true", dest="all", default=False)
    parser.add_option('-i', '--interval', dest='interval', help='interval for sampling points for plotting', metavar='INTERVAL', default=1)
    options, args = parser.parse_args()

    return options, args

def main():
    options, args = getOptions()
    if options.all:
        plot_all_pickles_in_directory(options)
    else:
        plot_pickle(options)

if __name__ == "__main__":
    main()
