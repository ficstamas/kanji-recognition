from utils.images import load_images
import numpy as np
from preprocessors.distribution import ccw_distribution
import tqdm
from skimage.measure import shannon_entropy
import skimage.feature as feature
import os
import random


def make_train_data(x: np.ndarray, y: np.ndarray):
    num_of_features = None
    dists = ccw_distribution(x, y)
    converted_values = []
    for i in tqdm.trange(dists.shape[0]):
        dist = dists[i].astype(np.float)
        image = np.ravel(x[i]).astype(np.float)
        entropy = shannon_entropy(x[i])
        canny = np.ravel(feature.canny(x[i])).astype(np.float)
        hrr, hrc, hcc = feature.hessian_matrix(x[i])
        hrr = np.ravel(hrr).astype(np.float)
        hrc = np.ravel(hrc).astype(np.float)
        hcc = np.ravel(hcc).astype(np.float)
        feature_vector = np.concatenate([np.array([entropy], dtype=np.float), dist, image, canny, hrr, hrc, hcc])
        del dist, image, entropy, canny, hrr, hrc, hcc
        indexed_features = {}
        for j in range(feature_vector.shape[0]):
            if feature_vector[j] != 0:
                indexed_features[j] = feature_vector[j]
        converted_values.append({"label": y[i], "data": indexed_features})
        if num_of_features is None:
            num_of_features = feature_vector.shape[0]
    return converted_values, num_of_features


def save_data(path: str, data: list, file_header: dict):
    random.seed(0)
    os_path = os.path.join(os.getcwd(), path)
    with open(os_path, mode='w', encoding='utf8') as f:
        f.write(file_header["entries"] + " " + file_header["features"] + " " + file_header["labels"] + "\n")
        while data.__len__() != 0:
            r = random.randint(0, data.__len__()-1)
            entry = data[r]
            f.write(str(entry["label"]))
            f.write(" ")
            for _features in entry["data"]:
                f.write(str(_features))
                f.write(":")
                f.write(str(entry["data"][_features]))
                f.write(" ")
            f.write("\n")
            data.remove(entry)


if __name__ == '__main__':
    kanjis = load_images(path="../../data/kkanji/kkanji2/", minimum_count=5, random_seed=0, category_limit=5)
    x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)
    train_converted, num_of_features = make_train_data(x_train, y_train)

    entries = str(x_train.shape[0])
    labels = str(np.unique(y_train).shape[0])
    features = str(num_of_features)

    header = {
        "entries": entries,
        "labels": labels,
        "features": features
    }

    save_data("../../data/slide_features_train.txt", train_converted, header)
    del x_train, y_train, train_converted, num_of_features, entries, labels, features, header

    test_converted, num_of_features = make_train_data(x_test, y_test)

    entries = str(x_test.shape[0])
    labels = str(np.unique(y_test).shape[0])
    features = str(num_of_features)

    header = {
        "entries": entries,
        "labels": labels,
        "features": features
    }

    save_data("../../data/slide_features_test.txt", test_converted, header)
