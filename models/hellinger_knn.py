from sklearn.neighbors import KNeighborsClassifier
from utils.images import load_images
from preprocessors.baseline import ravel_data
from sklearn import metrics
import logging
from utils.distances import hellinger
from preprocessors.density_function import classes_to_density_functions, sample_density_functions

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

kanjis = load_images(minimum_count=5, random_seed=0, category_limit=5)
x, y, _, _ = kanjis.train_test_split(None)
kdes = classes_to_density_functions(x, y)
del x, y

x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)

logging.info("Sampling train...")
x_train_prob = sample_density_functions(x_train, y_train, kdes)
logging.info("Sampling test...")
x_test_prob = sample_density_functions(x_test, y_test, kdes)

neigh = KNeighborsClassifier(n_neighbors=3, metric=hellinger)
neigh.fit(x_train_prob, y_train)

y_pred = neigh.predict(x_test_prob)

# train_acc = metrics.cohen_kappa_score(x_train, y_train)
test_acc = metrics.cohen_kappa_score(y_test, y_pred)

# logging.info(f"Train accuracy: {train_acc}")
logging.info(f"Test accuracy: {test_acc}")