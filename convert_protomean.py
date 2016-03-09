import caffe
import numpy as np
import sys

if len(sys.argv) != 2:
  print "Usage: python convert_protomean.py image.binaryproto"
  sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
print "Mean pixel value: %f" % np.mean(out)
