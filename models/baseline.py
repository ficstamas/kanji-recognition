from sklearn.neighbors import KNeighborsClassifier
from utils.images import load_images
from preprocessors.baseline import ravel_data
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import logging
import json
import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

kanjis = load_images(minimum_count=5, random_seed=0, category_limit=None)
x_train, y_train, x_test, y_test = kanjis.train_test_split(None)

parameters = [{"n_neighbors": [3, 4],
               "leaf_size": [30]}]

neigh = KNeighborsClassifier()
gs = GridSearchCV(neigh, parameters, cv=5, scoring=metrics.make_scorer(metrics.cohen_kappa_score), return_train_score=True, refit=True, verbose=2, n_jobs=5)
gs.fit(ravel_data(x_train), y_train)

logging.info(f"Best params: {gs.best_params_}")

val_means = gs.cv_results_['mean_test_score']
val_stds = gs.cv_results_['std_test_score']
val_params = gs.cv_results_['params']

wrap = {"means": val_means.tolist(),
        "stds": val_stds.tolist(),
        "params": val_params}

f = open("../results/baseline/results.json", mode="w", encoding="utf8")
json.dump(wrap, f)
f.close()

f = open("../results/baseline/detailed_results.json", mode="w", encoding="utf8")
out = {}
for key in gs.cv_results_:
    if type(gs.cv_results_[key]) is np.ndarray or type(gs.cv_results_[key]) is np.ma.MaskedArray:
        out[key] = gs.cv_results_[key].tolist()
    else:
        out[key] = gs.cv_results_[key]

json.dump(out, f)
f.close()

for mean, std, params in zip(val_means, val_stds, val_params):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
