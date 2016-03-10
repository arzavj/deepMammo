import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

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

def main():
    for f in os.listdir(pickles_dir):
        filename = os.path.join(pickles_dir, f)
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                train_loss, train_acc, val_loss, val_acc = pickle.load(f)
                filename_without_extension = os.path.splitext(filename)[0].split("/")[1]
                plot_loss(train_loss['pretrained'], filename_without_extension)
                plot_accuracy(train_acc['pretrained'], val_acc['pretrained'], filename_without_extension)

if __name__ == "__main__":
    main()