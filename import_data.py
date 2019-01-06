import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
from librosa import display
import tensorflow as tf
import os
import cv2
from PIL import Image


plt.ioff()
plt.switch_backend('Agg')


# WIDTH_RATIO = 5
# HEIGHT_RATIO = 3
# DPI = 96
IMAGE_WIDTH = 387
IMAGE_HEIGHT = 231
CHANNELS = 3
WORD_COUNT = 65
WORD_COUNT_W_A = 68
# N_CLASS = 0
# TRAIN_SIZE = 0
# TEST_SIZE = 0
# N_MELS = 0
# T_FRAMES = 0
#
#
# def get_n_mels():
#     return N_MELS
#
#
# def get_t_frames():
#     return T_FRAMES
#
#
# def set_n_mels(nmels):
#     N_MELS = nmels
#
#
# def set_t_frames(tframes):
#     T_FRAMES = tframes
#
#
# def get_width():
#     biggest_mat = get_t_frames()*get_n_mels()
#     inches = biggest_mat/DPI
#     width_in_inch = inches/HEIGHT_RATIO
#     IMAGE_WIDTH = width_in_inch*DPI
#     return width_in_inch
#
#
# def get_height():
#     biggest_mat = get_t_frames()*get_n_mels()
#     inches = biggest_mat/DPI
#     height_in_inch = inches/WIDTH_RATIO
#     IMAGE_HEIGHT = height_in_inch*DPI
#     return height_in_inch
#
#
# def get_train_size():
#     return TRAIN_SIZE
#
#
# def get_test_size():
#     return TEST_SIZE
#
#
# def get_classes():
#     return N_CLASS

#
# class dataset:
#     def __init__(self):
#         self.width = IMAGE_WIDTH
#         self.height = IMAGE_HEIGHT
#         self.train_x_list = []
#         self.train_y_list = []
#         self.test_x_list = []
#         self.test_y_list = []
#         self.img_list = []
#         self.label_list = []
#         self.freq_dict = {}
#         self.num_classes = 0
#         self.labels_keys = {}
#         self.n_inputs = self.width*self.height*CHANNELS
#         self.batch_counter = 0
#
#     def create_train_and_test_dataset_lists(self, path_of_images):
#         path_of_images = os.getcwd() + path_of_images
#         # a list of images path
#         # a list of their corresponding labels
#         # a (frequency) dictionary with number of each label
#         self.img_list, self.label_list, self.freq_dict = d.import_image_files(path_of_images)
#
#         # the number of classes for classification
#         # a dictionary with the numerical value of each class
#         self.num_classes, self.labels_keys = d.get_class_number_and_key_dict(self.freq_dict, threshold=40)
#
#         # lists of train dataset and test dataset images path
#         # the corresponding lists of labels for the train and test datasets
#         self.train_x_list, self.train_y_list, self.test_x_list, self.test_y_list = d.set_train_n_test(self.img_list,
#                                                                                                       self.label_list,
#                                                                                                       self.freq_dict,
#                                                                                                       self.labels_keys,
#                                                                                                       test_size=10)


def get_width():
    return IMAGE_WIDTH


def get_height():
    return IMAGE_HEIGHT


def set_width(new_width):
    IMAGE_WIDTH = new_width


def set_height(new_height):
    IMAGE_HEIGHT = new_height


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

    # temp_n_mels, temp_t_frames = mfccs.shape

    # if temp_n_mels > get_n_mels():
    #     # set_n_mels(temp_n_mels)
    #     N_MELS = temp_n_mels
    #
    # if temp_t_frames > get_t_frames():
    #     set_t_frames(temp_t_frames)

    return mfccs


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
    temp_n_mels, temp_t_frames = mspec.shape

    # if temp_n_mels > get_n_mels():
    #     set_n_mels(temp_n_mels)
    #
    # if temp_t_frames > get_t_frames():
    #     set_t_frames(temp_t_frames)

    return mspec


def featured_data_to_image(data, width, height, out_filename, mfcc):
    """
    plots the data to image and saves it as jpg file
    1 inch = 96px
    :param data: np.ndarray of the data to be saved as a file
    :param width: of image to save in inches
    :param height: of image to save in inches
    :param out_filename: filename to be saved
    :return: path to jpg file
    """

    plt.figure(figsize=(width, height), frameon=False) # 5(in) x 3(in) = 480(px) x 288(px)

    plt.rcParams['savefig.pad_inches'] = 0

    if mfcc:
        ax = librosa.display.specshow(data)
    else:
        ax = librosa.display.specshow(librosa.power_to_db(data, ref=np.max))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    plt.savefig(out_filename, bbox_inches='tight', pad_inches=0)
    print("saved image to: ", out_filename)
    plt.clf()
    plt.close()


def mp3_to_jpg(filename, mfcc=True, width=480, height = 288, width_inch=5, height_inch=3, dpi=96, out_folder="jpg"):
    if mfcc:
        arr = mp3_to_mfcc(filename)
    else:
        arr = mp3_to_spectrogram(filename)
    fn_split = filename.split("/")
    out_path = filename.replace(fn_split[-1], out_folder)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_path = out_path + "/" + fn_split[-1][:-4]
    out_path = out_path + ".jpg"
    featured_data_to_image(arr, width_inch, height_inch, out_path, mfcc)


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
    if len(temp_image_list) < 1:
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


def get_class_number_and_key_dict(freq_dict, threshold=40):
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

    # N_CLASS = result
    print("Keys and Values are:")

    for k, v in labels_key.items():
        print(k, " :", v)

    return result, labels_key


def set_train_n_test(image_list, label_list, freq_dict, key_dict, test_size=10):
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

    # TEST_SIZE = len(test_labels)
    # TRAIN_SIZE = len(train_labels)

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


def load_image(path_to_image):
    decoded_img = cv2.imread(path_to_image)
    return decoded_img


# def resize_image(decoded_img):
#     decoded_img.set_shape([None, None, None])
#     resized_img = tf.image.resize_images(decoded_img, [IMAGE_WIDTH, IMAGE_HEIGHT])
#     return resized_img


def reshape_decoded_image(decoded_image):
    return np.reshape(decoded_image, IMAGE_WIDTH*IMAGE_HEIGHT*CHANNELS)

# def reshape_decoded_image(decoded_image):
#     return tf.reshape(decoded_image, IMAGE_WIDTH*IMAGE_HEIGHT*CHANNELS)

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