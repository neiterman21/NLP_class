import glob
import csv
import numpy as np # linear algebra
import os


def parst_data_labels(file):
    csv_file = open(file,'r')
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    data_labels = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            recording = {
                "age"           : row[0],
                "age_onset"     : row[1] ,
                "birthplace"    : row[2] ,
                "filename"      : row[3] ,
                "native_language" : row[4],
                "sex"           : row[5] ,
                "speakerid"     : row[6] ,
                "country"       : row[7] ,
                "file_missing"  : row[8]
            }
            data_labels.append(recording)
            line_count += 1
    print(f'Processed {line_count} lines.')
    return data_labels


def read_labeld_image_list() :
    path = os.getcwd()
    labels_raw = parst_data_labels(path + "/data/speakers_all.csv")
    image_list = glob.glob(path + "/data/melspectogram/" + "*.jpg")
    labels = []
    for image in image_list:
        image_name = image.split("/")[-1].split(".jpg")[0]
        for l in labels_raw:
            if image_name == l["filename"] and "FALSE" in l['file_missing'] :
                labels.append(l["native_language"])
                break
    return np.array(image_list) , np.array(labels)

