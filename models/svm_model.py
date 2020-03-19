from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from utils.images import load_images
from preprocessors.baseline import ravel_data

kanjis = load_images('C:\\Users\\takac\\OneDrive\\Desktop\\asd',
                     minimum_count=5, random_seed=0, category_limit=None)
x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)

model = LinearSVC()
model.fit(ravel_data(x_train), y_train)

predictions = model.predict(ravel_data(x_test))
print(classification_report(y_test, predictions))