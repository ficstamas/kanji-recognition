from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from utils.images import load_images
from sklearn import metrics
import logging
from preprocessors.baseline import ravel_data

kanjis = load_images('C:\\Users\\HealthTeam\\Desktop\\kanji\\kkanji\\kkanji2',
                     minimum_count=5, random_seed=0, category_limit=None)
x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)

model = LinearSVC()
model.fit(ravel_data(x_train), y_train)

predictions = model.predict(ravel_data(x_test))
test_acc = metrics.cohen_kappa_score(y_test, predictions)

print('Accuracy Score:', metrics.accuracy_score(y_test,predictions))
logging.info(f"Test accuracy: {test_acc}")

file = open('../result/svm/acc.txt','w')
file.write(str(test_acc))

#print(classification_report(y_test, predictions))