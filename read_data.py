import glob
import csv
import numpy as np  # linear algebra
import os


def parst_data_labels(file):
    """
    method for parsing /data/speakers_all.csv file
    :param file: path to speakers_all.csv
    :return: a list with labels
    """
    csv_file = open(file, 'r')
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    data_labels = []
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            recording = {
                "age": row[0],
                "age_onset": row[1],
                "birthplace": row[2],
                "filename": row[3],
                "native_language": row[4],
                "sex": row[5],
                "speakerid": row[6],
                "country": row[7],
                "file_missing": row[8]
            }
            data_labels.append(recording)
            line_count += 1
    print(f'Processed {line_count} lines.')
    return data_labels


def read_labeld_image_list():
    """
    looks for images dataset of type jpg (formatted: labelXXX.jpg (XXX number))
    matches labels from speakers_all.csv
    :return: numpy array of path to images and their corresponding label
    """
    path = os.getcwd()
    labels_raw = parst_data_labels(path + "/data/speakers_all.csv")
    image_list = glob.glob(path + "/data/melspectogram/" + "*.jpg")
    labels = []
    for image in image_list:
        image_name = image.split("/")[-1].split(".jpg")[0]
        for l in labels_raw:
            if image_name == l["filename"] and "FALSE" in l['file_missing']:
                labels.append(l["native_language"])
                break
    return np.array(image_list), np.array(labels)


def get_image_and_label():
    """
    creates a filtered dataset with 6 classes of audio recordings
    which contains at least 48 samples
    :return: lists of:
                paths to filtered images
                unfiltered labels
                filtered labels
                numeric labels of filtered labels
    """
    image_list_, label_list_ = read_labeld_image_list()

    image_list = []
    label_list = []

    for path, lable in zip(image_list_, label_list_):
        if "english" in lable or "spanish" in lable or "arabic" in lable or "mandarin" in lable or "french" in lable or "russian" in lable:
            image_list.append(path)
            label_list.append(lable)
    label_names = list(set(label_list))

    numeric_labels = []
    for l in label_list:
        numeric_labels.append(label_names.index(l))

    return image_list, label_list, label_names, numeric_labels
