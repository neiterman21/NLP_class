import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
from librosa import display
import tensorflow as tf


IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
N_CLASS = 0


def get_classes():
    return N_CLASS


def mp3_to_mfcc(file):
    """
    converts mp3 file to the representing Mel-Frequency Cepstral Coefficients

    Wikipedia https://en.wikipedia.org/wiki/Mel-frequency_cepstrum :
    " the mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound,
      based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency...
      ...the frequency bands are equally spaced on the mel scale,
      which approximates the human auditory system's response more closely
      than the linearly-spaced frequency bands used in the normal cepstrum."

    :param file: path to mp3 file
    # :param label: corresponding label (of mp3 file) to be saved as file name
    :return: MFCC sequence type: np.ndarray
    """
    y, sr = librosa.load(file, mono=False)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return mfccs[0]


def mp3_to_spectrogram(file):
    """
    converts mp3 file to the representing Mel-Spectrogram

    "represents an acoustic time-frequency representation of a sound:
     the power spectral density P(f, t).
     It is sampled into a number of points around equally spaced times ti and frequencies fj
     (on a Mel frequency scale)."

    :param file: path to mp3 file
    # :param label: corresponding label (of mp3 file) to be saved as file name
    :return: Mel-Spectrogram sequence type: np.ndarray
    """
    y, sr = librosa.load(file, mono=False)
    mspec = librosa.feature.melspectrogram(y=y, sr=sr)
    return mspec[0]


def featured_data_to_image(data, out_filename):
    """
    plots the data to image and saves it as jpg file
    :param data: np.ndarray of the data to be saved as a file
    :param out_filename: filename to be saved
    :return: path to jpg file
    """
    plt.figure(figsize=(4, 3), frameon=False)
    plt.rcParams['savefig.pad_inches'] = 0
    ax = librosa.display.specshow(data)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    plt.savefig(out_filename)
    # plt.savefig(out_folder + out_filename)


def import_image_files(path):
    """
    finds all image files with their labels in the directory
    extracts labels from image files assuming pattern "label#.jpg OR label#.png"
    :param path: to directory where image files are
    :return: image files list     (image_list) - contains path to image files
             label's list         (label_list) - contains the corresponding label for each image
             frequency dictionary (frequ_dict) - contains the amount(value) of samples per label (key)
    """
    if not path.endswith("/"):
        path = path + "/"

    temp_image_list = glob.glob(path + "*.jpg")
    ext = "*.jpg"
    if temp_image_list.count() < 1:
        temp_image_list = glob.glob(path + "*.png")
        ext = "*.png"

    image_list = []
    label_list = []
    frequ_dict = {}

    for image_path in temp_image_list:
        image_name = image_path.split("/")[-1].split(ext)[0]
        label = ''.join(i for i in image_name if not i.isdigit())
        image_list.append(image_path)
        label_list.append(label)

    unique, count = np.unique(label_list, return_counts=True)
    for _ in range(count.size):
        frequ_dict[unique[_]] = count[_]

    return image_list, label_list, frequ_dict


def get_class_number_and_key_dict(freq_dict, threshold):
    """
    calculates the amount of classification classes based on a minimum threshold of samples per class
    and creates a dictionary with key,value corresponds to label,index
    :param freq_dict: frequency dictionary returned by import_image_files(path)
    :param threshold: minimum amount of samples per class
    :return: result = number of classes
             labels_key = (labels,index) dictionary
    """
    labels_key = {}
    key_index = 0
    result = 0
    for label in freq_dict:
        if freq_dict[label] >= threshold:
            result = result + 1
            labels_key[label] = key_index
            key_index = key_index + 1

    N_CLASS = result
    print("Keys and Values are:")

    for k, v in labels_key.items():
        print(k, " :", v)

    return result, labels_key


def set_train_n_test(image_list, label_list, freq_dict, key_dict, test_size = 10 ):
    """
    creates a train and test datasets
    only labels in key dictionary (key_dict) gets into the dataset

    :param image_list: list of image files
    :param label_list: list of corresponding labels
    :param freq_dict: a dictionary contains the amount of samples per label
    :param key_dict: valid labels (passed threshold test) with corresponding numerical label
    :param test_size: the amount of test samples per label (10% default)
    :return: train_images
             train_labels
             test_images
             test_labels
    """
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    test_size_for_sample = {}

    for label in key_dict:
        test_size_for_sample[label] = freq_dict[label] // test_size

    for _ in range(len(label_list)):
        label = label_list[_]
        if label in key_dict:
            if test_size_for_sample[label] > 0:
                test_images.append(image_list[_])
                test_labels.append(label)
                test_size_for_sample[label] = test_size_for_sample[label] - 1
            else:
                train_images.append(image_list[_])
                train_labels.append(label)

    return train_images, train_labels, test_images, test_labels


def parse_image(filename, label):
    """
    Parse the image to a fixed size with width (IMAGE_WIDTH = 800) and height (IMAGE_HEIGHT = 600)
    example in https://www.tensorflow.org/guide/datasets#preprocessing_data_with_datasetmap
    :param filename: jpg file path list
    :param label: labels list
    :return:    label corresponding to image
                image_resized to the correct size
    """
    image_source = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_source)
    # image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [IMAGE_WIDTH, IMAGE_HEIGHT])
    return label, image_resized


# def create_tf_dataset(filenames_list, labels_list):
#     """
#
#     :param filenames_list:
#     :param labels_list:
#     :return:
#     """
#     filenames = tf.constant(filenames_list)
#     labels = tf.constant(labels_list)
#     dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
#     dataset = dataset.map(parse_image)
#
#     return dataset
