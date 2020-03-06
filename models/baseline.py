from sklearn.neighbors import KNeighborsClassifier
from utils.images import load_images
from preprocessors.baseline import ravel_data
from sklearn import metrics

kanjis = load_images(minimum_count=5, random_seed=0, category_limit=None)
x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)

neigh = KNeighborsClassifier(n_neighbors=3, n_jobs=3)
neigh.fit(ravel_data(x_train), y_train)

y_pred = neigh.predict(ravel_data(x_test))

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)