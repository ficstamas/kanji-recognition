import cv2 as cv
import os
import numpy as np
import tqdm
import logging
import random
import math

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class Kanjis:
    """
    Load and stores Kanjis
    """

    _images = np.array([np.zeros([64,64], dtype=np.uint8)], dtype=np.uint8)
    _labels = np.array([0], dtype=np.uint)

    # label to index
    l2i = {}
    # index to label
    i2l = {}

    # Change these carefully
    _id = 0
    first_image = True

    def __init__(self, seed=None):
        """

        :param seed: Seed for random number generation
        """
        random.seed = seed

    def add_label(self, label: str):
        """
        Add label, it will create conversion dictionary as well
        :param label:
        :return:
        """
        if label not in self.l2i:
            self.l2i[label] = self._id
            self.i2l[self._id] = label
            self._id += 1

    def add_image(self, img: np.ndarray, label: str):
        """
        Add a single image
        :param img: [64, 64] image
        :param label: label
        :return:
        """
        if self.first_image:
            self._images[0, :, :] = img
            self._labels[0] = self.l2i[label]
            self.first_image = False
        else:
            self._images = np.concatenate([self._images, [img]], axis=0)
            self._labels = np.concatenate([self._labels, [self.l2i[label]]], axis=0)

    def add_images(self, imgs: np.ndarray, label: str):
        """
        Add multiple images with same label
        :param imgs: [*, 64, 64] images
        :param label: class label
        :return:
        """
        if self.first_image:
            self._images = imgs
            self._labels = np.array([self.l2i[label]]*imgs.shape[0], dtype=np.uint)
            self.first_image = False
        else:
            self._images = np.concatenate([self._images, imgs], axis=0)
            self._labels = np.concatenate([self._labels, [self.l2i[label]]*imgs.shape[0]], axis=0)

    def train_test_split(self, train_ratio=0.6):
        """
        Creates train in test splits
        :param train_ratio:
        :return:
        """
        mask = np.zeros(self._labels.shape, dtype=np.bool)
        size = mask.shape[0]

        # from every class keeping a percent (determined by train_percent) of the data in train
        for label in self.i2l:
            samples = np.where(self._labels == label)[0]
            random_vector = random.sample(samples.tolist(), math.ceil(samples.shape[0] * train_ratio))
            mask[random_vector] = 1
        return self._images[mask], self._labels[mask], self._images[~mask], self._labels[~mask]

    def __len__(self):
        return self._images.shape[0]

    def __str__(self):
        return f"{self.__len__()} images\n {self.l2i.__len__()} classes\n"

    def __repr__(self):
        return self.__str__()


def load_images(path="../data/kkanji/kkanji2/", category_limit=None, minimum_count=5, random_seed=None) -> Kanjis:
    """
    Load images and labels into object
    :param path: Path to the folder of the Kanjis
    :param category_limit: Maximum number of categories to load
    :param random_seed: Seed for random number generation
    :return:
    """

    # init Kanjis object
    kanjis = Kanjis(seed=random_seed)

    # creating path
    path = os.path.join(os.curdir, path)

    logging.info(f"Loading images from {path}...")
    num = 1
    # walk in the folders
    for root, dirs, files in tqdm.tqdm(list(os.walk(path))):
        # skip if it is not containing files (just th first entry satisfies that)
        if files.__len__() == 0:
            continue
        # getting label from folder name
        label = str(root.split("/")[-1])

        images = np.array([np.zeros([64, 64], dtype=np.uint8)], dtype=np.uint8)
        # loading files
        for file in files:
            img = cv.imread(os.path.join(root, file), cv.IMREAD_GRAYSCALE)
            images = np.concatenate([images, [img]], axis=0)
        # checking if the class has enough samples
        if images.shape[0] >= minimum_count:
            # adding label
            kanjis.add_label(label)
            kanjis.add_images(images, label)
            # limiting loaded categories
            if category_limit is not None and num == category_limit:
                break
            elif category_limit is not None:
                num += 1

    logging.info(f"{kanjis.__len__()} kanji under {kanjis.l2i.__len__()} classes is loaded")
    return kanjis
