import sys
import numpy as np
import lmdb
import caffe
from collections import defaultdict
from optparse import OptionParser
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

caffe.set_device(0)
caffe.set_mode_gpu()

val_lmdb = "/home/ubuntu/2x_padding_224x224_dataset/mass_2x_padding_dataset_val_lmdb"

# Extract mean from the mean image file
def get_mean_image(options):
    mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
    f = open(options.mean_file_binaryproto, 'rb')
    mean_blobproto_new.ParseFromString(f.read())
    mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
    f.close()
    return mean_image

def get_true_pred_labels(options, net, mean_image):
    y_true, y_pred = [], []
    lmdb_env = lmdb.open(val_lmdb)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.astype(np.uint8)
        out = net.forward_all(data=np.asarray([image]) - mean_image)
        plabel = int(out['prob'][0].argmax(axis=0))
        y_true.append(label)
        y_pred.append(plabel)

    return y_true, y_pred

def getOptions():
    '''
    Get command-line options and handle errors.
    :return: command-line options and arguments
    '''
    parser = OptionParser()
    parser.add_option('-m', '--mean', dest='mean_file_binaryproto', default="/home/ubuntu/2x_padding_224x224_dataset/mass_2x_padding_dataset_train_mean.binaryproto",
                      help='path name to mean binary proto', metavar='MEAN_PATH')
    parser.add_option('-w', '--weights', dest='weights_path', default="/home/ubuntu/caffe/models/deep_mammo/bvlc_alexnet.caffemodel",
                      help='path name to trained weights caffemodel', metavar='WEIGHTS')
    parser.add_option('-d', '--deploy', dest='deploy_prototxt_file_path', default="/home/ubuntu/caffe/models/deep_mammo/alex_deploy.prototxt",
                      help='path name to deploy prototoxt', metavar='DEPLOY')
    options, args = parser.parse_args()

    return options, args

def main():
    options, args = getOptions()

    # CNN reconstruction and loading the trained weights
    net = caffe.Net(options.deploy_prototxt_file_path, options.weights_path, caffe.TEST)
    mean_image = get_mean_image(options)
    y_true, y_pred = get_true_pred_labels(options, net, mean_image)
    print "Precision: %f" % precision_score(y_true, y_pred)
    print "Recall: %f" % recall_score(y_true, y_pred)
    print "Accuracy: %f" % accuracy_score(y_true, y_pred)

if __name__ == '__main__':
    main()