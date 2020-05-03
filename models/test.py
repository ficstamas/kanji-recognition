from preprocessors.feature_extraction.skeleton import barmi, define_graph, corner_points, skeleton
#from utils.images import load_images
import cv2 as cv
from sklearn.naive_bayes import GaussianNB
from utils.images import load_images
from sklearn import metrics
import logging
import numpy as np
from preprocessors.baseline import ravel_data
from sklearn.svm import LinearSVC


kanjis = load_images('C:\\Users\\HealthTeam\\Desktop\\mini-kanji', minimum_count=10, random_seed=0, category_limit=None)
x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)

#img = x_train[2]

#S = skeleton(img)
num = 0
m = []
for img in x_train:
    C = define_graph(img)
    m.append(C)
    num+=1
    print(num)
m_test=[]
for img in x_test:
    C = define_graph(img)
    m_test.append(C)

x_train = np.array(m)
print(x_train.shape)
x_test = np.array(m_test)
#asd = define_graph(img)
print('tanítás')

model = LinearSVC()
model.fit(ravel_data(x_train), y_train)

predictions = model.predict(ravel_data(x_test))
test_acc = metrics.cohen_kappa_score(y_test, predictions)

print('Accuracy Score:', metrics.accuracy_score(y_test,predictions))
logging.info(f"Test accuracy: {test_acc}")

