import os
from sklearn.model_selection import GridSearchCV
import logging
import json
import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def save(gs: GridSearchCV, path: str):
    """
    Saves training results
    :param gs: GridSearchCV object reference after fitting
    :param path: Path to the directory
    :return:
    """
    logging.info(f"Best params: {gs.best_params_}")

    val_means = gs.cv_results_['mean_test_score']
    val_stds = gs.cv_results_['std_test_score']
    val_params = gs.cv_results_['params']

    wrap = {"means": val_means.tolist(),
            "stds": val_stds.tolist(),
            "params": val_params}

    f = open(os.path.join(path, "results.json"), mode="w", encoding="utf8")
    json.dump(wrap, f)
    f.close()

    f = open(os.path.join(path, "detailed_results.json"), mode="w", encoding="utf8")
    out = {}
    for key in gs.cv_results_:
        if type(gs.cv_results_[key]) is np.ndarray or type(gs.cv_results_[key]) is np.ma.MaskedArray:
            out[key] = gs.cv_results_[key].tolist()
        else:
            out[key] = gs.cv_results_[key]

    json.dump(out, f)
    f.close()

    for mean, std, params in zip(val_means, val_stds, val_params):
        logging.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
