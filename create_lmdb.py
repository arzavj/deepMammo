import os
import sys

height = 224
width = 224

def main(args):
  if args[1].endswith("/"):
    args[1] = args[1][:-1]
  create_lmdb(args[1] + "_train")
  # create_lmdb(args[1] + "_val")
  # create_lmdb(args[1] + "_test")

def create_lmdb(dataset_name):
  command = "GLOG_logtostderr=1 ./caffe/build/tools/convert_imageset --resize_height=%d --resize_width=%d %s/ /mnt/%s_labels.txt /mnt/%s_lmdb" % (height, width, dataset_name, dataset_name, dataset_name)
  print command
  os.system(command)

# need to pass "mass_50_padding_dataset" as arg
if __name__ == "__main__":
  main(sys.argv)
