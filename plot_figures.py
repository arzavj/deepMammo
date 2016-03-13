import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os
from optparse import OptionParser

pickles_dir = "pickles"
figures_dir = "figures"

def plot_loss(loss, filename):
    plt.plot(loss.T)
    plt.xlabel('Iteration #')
    plt.ylabel('Training Loss')
    plt.grid()
    plt.savefig('%s/loss_curves_%s.png' % (figures_dir, filename))
    plt.close()

def plot_accuracy(train_acc, val_acc, filename):
    plt.plot(train_acc.T)
    plt.plot(val_acc.T)
    plt.xlabel('Iteration #')
    plt.ylabel('Accuracy')
    plt.legend(['Train Accuracy', 'Val Accuracy'], loc=4)
    plt.grid()
    plt.savefig('%s/accuracy_curves_%s.png' % (figures_dir, filename))
    plt.close()

def plot_all_pickles_in_directory():
    for f in os.listdir(pickles_dir):
        filename = os.path.join(pickles_dir, f)
        if os.path.isfile(filename):
            plot_pickle(filename)

def plot_pickle(filename):
    with open(filename, 'r') as f:
        train_loss, train_acc, val_loss, val_acc = pickle.load(f)
        filename_without_extension = os.path.splitext(filename)[0].split("/")[1]
        plot_loss(train_loss['pretrained'], filename_without_extension)
        plot_accuracy(train_acc['pretrained'], val_acc['pretrained'], filename_without_extension)

def getOptions():
    '''
    Get command-line options and handle errors.
    :return: command-line options and arguments
    '''
    parser = OptionParser()
    parser.add_option('-p', '--pickle', dest='pickle_filename',
                      help='path name to pickle file', metavar='PICKLE')
    parser.add_option('-a', action="store_true", dest="all", default=False)
    options, args = parser.parse_args()

    return options, args

def main():
    options, args = getOptions()
    if options.all:
        plot_all_pickles_in_directory()
    else:
        plot_pickle(options.pickle_filename)

if __name__ == "__main__":
    main()