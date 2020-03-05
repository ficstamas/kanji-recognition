import cv2 as cv
import os
import numpy as np
import tqdm
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class Kanjis:
    """
    Load and stores Kanjis
    """
    _images = np.array([np.zeros([64,64], dtype=np.uint8)], dtype=np.uint8)
    _labels = np.array([0], dtype=np.uint)
    l2i = {}
    i2l = {}
    _id = 0
    first_image = True

    def __init__(self):
        pass

    def add_label(self, label: str):
        if label not in self.l2i:
            self.l2i[label] = self._id
            self.i2l[self._id] = label
            self._id += 1

    def add_image(self, img: np.ndarray, label: str):
        if self.first_image:
            self._images[0, :, :] = img
            self._labels[0] = self.l2i[label]
            self.first_image = False
        else:
            self._images = np.concatenate([self._images, [img]], axis=0)
            self._labels = np.concatenate([self._labels, [self.l2i[label]]], axis=0)

    def __len__(self):
        return self._images.shape[0]

    def __str__(self):
        return f"{self.__len__()} images\n {self.l2i.__len__()} classes\n"

    def __repr__(self):
        return self.__str__()


def load_images(path="../data/kkanji/kkanji2/", limit=None) -> Kanjis:
    """
    Load images and labels into object
    :param path: Path to the folder of the Kanjis
    :param limit: Maximum number of categories to load
    :return:
    """

    # init Kanjis object
    kanjis = Kanjis()

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
        # adding label
        kanjis.add_label(label)
        # loading files
        for file in files:
            img = cv.imread(os.path.join(root, file), cv.IMREAD_GRAYSCALE)
            kanjis.add_image(img, label)
        # limiting loaded categories
        if limit is not None and num == limit:
            break
        elif limit is not None:
            num += 1

    logging.info(f"{kanjis.__len__()} kanji under {kanjis.l2i.__len__()} classes is loaded")
    return kanjis
