from sklearn.neighbors import KNeighborsClassifier
from utils.images import load_images
from preprocessors.baseline import ravel_data
from sklearn import metrics
import logging
from utils.distances import hellinger

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

kanjis = load_images(minimum_count=5, random_seed=0, category_limit=None)
x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)

neigh = KNeighborsClassifier(n_neighbors=3, metric=hellinger)
neigh.fit(ravel_data(x_train), y_train)

y_pred = neigh.predict(ravel_data(x_test))

# train_acc = metrics.cohen_kappa_score(x_train, y_train)
test_acc = metrics.cohen_kappa_score(y_test, y_pred)

# logging.info(f"Train accuracy: {train_acc}")
logging.info(f"Test accuracy: {test_acc}")