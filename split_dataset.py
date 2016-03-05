import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import sys

padding = 50
df = pd.read_csv('mass_case_description.csv')
pathology_to_label = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}
original_dataset_dir = "mass_50_padding_dataset"
train_output_dataset_dir = "mass_%d_padding_dataset_train" % padding
val_output_dataset_dir = "mass_%d_padding_dataset_val" % padding
test_output_dataset_dir = "mass_%d_padding_dataset_test" % padding
train_fraction = 0.8
val_fraction = 0.1
test_fraction = 1.0 - train_fraction - val_fraction


def get_filenames():
    benign_filenames = []
    malignant_filenames = []
    for f in listdir(original_dataset_dir):
        if isfile(join(original_dataset_dir, f)):
            filename_without_extension = os.path.splitext(f)[0]
            filename_parts = filename_without_extension.split("_")
            if (filename_parts[0] == ".DS"): continue
            patient_id = "_".join(filename_parts[0:2])
            side = filename_parts[2]
            view = filename_parts[3]
            label = pathology_to_label[df[df['patient_id'] == patient_id][df['side'] == side][df['view'] == view]['pathology'].values[0]]
            if label == pathology_to_label['BENIGN']:
                benign_filenames.append(f)
            else:
                malignant_filenames.append(f)
    return benign_filenames, malignant_filenames

def create_dir_if_not_exists(output_dataset_dir):
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)

def move_files(filenames, output_dir):
    for f in filenames:
        infilename = join(original_dataset_dir, f)
        outfilename = join(output_dir, f)
        os.rename(infilename, outfilename)

def main(args):
    create_dir_if_not_exists(train_output_dataset_dir)
    create_dir_if_not_exists(val_output_dataset_dir)
    create_dir_if_not_exists(test_output_dataset_dir)
    benign_filenames, malignant_filenames = get_filenames()
    num_examples = len(benign_filenames) + len(malignant_filenames)
    num_val_benign = int(0.5 * round(val_fraction * num_examples))
    num_test_benign = int(0.5 * round(test_fraction * num_examples))
    val_benign_names = benign_filenames[:num_val_benign]
    val_malignant_names = malignant_filenames[:num_val_benign]
    del benign_filenames[:num_val_benign]
    del malignant_filenames[:num_val_benign]
    test_benign_names = benign_filenames[:num_test_benign]
    test_mal_names = malignant_filenames[:num_test_benign]
    del benign_filenames[:num_test_benign]
    del malignant_filenames[:num_test_benign]
    move_files(val_benign_names, val_output_dataset_dir)
    move_files(val_malignant_names, val_output_dataset_dir)
    move_files(test_benign_names, test_output_dataset_dir)
    move_files(test_mal_names, test_output_dataset_dir)
    move_files(benign_filenames, train_output_dataset_dir)
    move_files(malignant_filenames, train_output_dataset_dir)


if __name__ == "__main__":
    main(sys.argv)