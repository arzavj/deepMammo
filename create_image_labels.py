import os
from os import listdir
from os.path import isfile, join
import pandas as pd

images_dir = "mass_50_padding_dataset"
df = pd.read_csv('mass_case_description.csv')
pathology_to_label = {'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}

def main():
    label_file = open(images_dir + "_labels.txt", "w")
    for f in listdir(images_dir):
        if isfile(join(images_dir, f)):
            filename_without_extension = os.path.splitext(f)[0]
            filename_parts = filename_without_extension.split("_")
            patient_id = "_".join(filename_parts[0:2])
            side = filename_parts[2]
            view = filename_parts[3]
            label = pathology_to_label[df[df['patient_id'] == patient_id][df['side'] == side][df['view'] == view]['pathology'].values[0]]
            label_file.write("%s %d\n" % (f, label))
    label_file.close()

if __name__ == "__main__":
    main()