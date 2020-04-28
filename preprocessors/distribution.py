import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def ccw_distribution(x: np.ndarray, y: np.ndarray):
    logging.info("Constructing distributions...")
    labels = np.unique(y)

    # class_dist = {}
    # for i in range(labels.shape[0]):
    #     _class = x[np.where(y == labels[i])]
    #     _class_dist = None
    #     for j in range(_class.shape[0]):
    #         margin = np.histogramdd(np.ravel(_class[j]), bins=256)[0] / _class[j].size
    #         margin = np.ravel(margin)
    #         if _class_dist is None:
    #             _class_dist = margin
    #         else:
    #             _class_dist = _class_dist + margin
    #     _class_dist = _class_dist / _class.shape[0]
    #     class_dist[i] = _class_dist

    dist = []

    for i in range(x.shape[0]):
        img = x[i]
        margin = np.histogramdd(np.ravel(img), bins=256)[0] / img.size
        margin = np.ravel(margin)
        # cat = class_dist[y[i]].tolist()
        # prob = np.zeros(cat.shape)
        # for j in range(margin.shape[0]):
        #     if cat[j] == 0:
        #         prob[j] = 0
        #         continue
        #     prob[j] = -margin[j]*np.log(cat[j])
        # for j in range(margin.shape[0]):
        #     prob[j] = np.exp(prob[j])/np.sum(np.exp(prob))
        dist.append(margin)
    logging.info("Constructing distributions: Done!")

    return np.array(dist)


def test_distribution(x: np.ndarray) -> np.ndarray:
    logging.info("Constructing test distributions...")
    dist = []
    for i in range(x.shape[0]):
        img = x[i]
        margin = np.histogramdd(np.ravel(img), bins=256)[0] / img.size
        margin = np.ravel(margin)
        dist.append(margin)
    logging.info("Constructing test distributions: Done!")
    return np.array(dist)