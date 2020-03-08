from sklearn.neighbors import KNeighborsClassifier
from utils.images import load_images
from preprocessors.baseline import ravel_data
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from utils.save.grid_search_cv import save as GSCVSave
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

kanjis = load_images(minimum_count=5, random_seed=0, category_limit=5)
x_train, y_train, _, _ = kanjis.train_test_split(None)

parameters = [{"n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               "leaf_size": [30]}]

neigh = KNeighborsClassifier()
gs = GridSearchCV(neigh, parameters, cv=5, scoring=metrics.make_scorer(metrics.cohen_kappa_score), return_train_score=True, refit=True, verbose=2, n_jobs=5)
gs.fit(ravel_data(x_train), y_train)

GSCVSave(gs, "../results/baseline/")
