#!/usr/local/bin python

import numpy as np # linear algebra
import cv2


# Image Parameters
N_CLASSES = 2 # CHANGE HERE, total number of classes
IMG_WIDTH = 535 # CHANGE HERE, the image height to be resized to
IMG_HEIGHT= 396 # CHANGE HERE, the image width to be resized to
CHANNELS = 3 # The 3 color channels, change to 1 if grayscale

class DataKeeper():
    def __init__(self, image_paths , labels, label_names):
        assert len(image_paths) == len(labels) , print("labels list len isn't equl to images paths len")
        self.image_paths = image_paths
        self.labels = labels
        self.label_names = label_names
        self._batch_size = len(labels)
        self._curent_index = 0


    def setBatchSize(self, batch_size):
        self._batch_size = batch_size

    def getNumOfBatches(self):
        return (int)(len(self.labels) / self._batch_size)

    def read_np_images(self,imagepath , lable):
        imTr = []
        lTr  = []
        for path , l in zip(imagepath , lable) :
            im = cv2.imread(path )
            imTr.append(cv2.resize(im, (IMG_WIDTH,IMG_HEIGHT), interpolation=cv2.INTER_CUBIC))
            lTr.append(np.float32(l))
        return   imTr ,  lTr

    def getNextBatch(self):
        data_image = self.image_paths[self._curent_index :  self._curent_index + self._batch_size]
        data_label = self.labels[self._curent_index :  self._curent_index + self._batch_size]
        imTr ,lTr = self.read_np_images(data_image , data_label)
        imTr = np.array(imTr, dtype='float32') #as mnist
        imTr = np.reshape(imTr,[imTr.shape[0],imTr.shape[1]*imTr.shape[2]*imTr.shape[3]])
        a = np.zeros((len(data_label),len(self.label_names)))
        for line , i in  zip(a, range(len(data_label))):
            line[data_label[i]] = 1
        lTr = a
        lTr = np.array(lTr,dtype='float64')

        return imTr ,lTr

