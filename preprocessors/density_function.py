import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import Normalizer, StandardScaler


def classes_to_density_functions(x, y):
    labels = np.unique(y)
    functions = {}
    for i in range(labels.shape[0]):
        _class = x[np.where(y == labels[i])]
        _class = _class/255

        _center0 = np.average(_class, axis=1)
        _center1 = np.average(_class, axis=2)
        _m_class = np.concatenate([_center0, _center1], axis=1)
        _p_class = np.average(_m_class, axis=0)
        # _p_class = StandardScaler().fit_transform(_p_class[np.newaxis, :])
        # _p_class = Normalizer(norm='l1').fit_transform(_p_class)
        kde = KernelDensity(bandwidth=0.2, kernel='cosine')
        kde.fit(_p_class[np.newaxis, :].reshape(-1, 1))
        functions[i] = kde
    return functions


def sample_density_functions(x: np.ndarray, y: np.ndarray, kdes: dict):
    data = []
    for i in range(y.shape[0]):
        img = x[i]/255
        # img = StandardScaler().fit_transform(img)
        # img = Normalizer(norm='l1').fit_transform(img)
        img = img.reshape(x.shape[1]*x.shape[2]).reshape(-1, 1)
        kde: KernelDensity
        kde = kdes[y[i]]
        p = kde.score_samples(img)
        p = np.exp(p)
        data.append(p)
    return np.array(data)



