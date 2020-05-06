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
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


kanjis = load_images('C:\\Users\\HealthTeam\\Desktop\\kanji\\kkanji\\kkanji2', minimum_count=10, random_seed=0, category_limit=None)
x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)

#img = x_train[2]

#S = skeleton(img)
num = 0
m = []
print('train_halmaz')
for img in x_train:
    C = define_graph(img)
    m.append(C)
    num+=1

print('test halmaz')
m_test=[]
for img in x_test:
    C = define_graph(img)
    m_test.append(C)

x_train = np.array(m)
x_test = np.array(m_test)
#asd = define_graph(img)

print('tanítás')
print('Naive')
gnb = GaussianNB()
gnb.fit(ravel_data(x_train), y_train)

pred = gnb.predict(ravel_data(x_test))

test_acc = metrics.cohen_kappa_score(y_test, pred)

print('Accuracy Score:', metrics.accuracy_score(y_test,pred))
logging.info(f"Test accuracy: {test_acc}")

file = open('../results/Naive_Bayes/acc_graph.txt','w')
file.write(str(test_acc))

print('SVM')
model = LinearSVC()
model.fit(ravel_data(x_train), y_train)

predictions = model.predict(ravel_data(x_test))
test_acc = metrics.cohen_kappa_score(y_test, predictions)

print('Accuracy Score:', metrics.accuracy_score(y_test,predictions))
logging.info(f"Test accuracy: {test_acc}")

file = open('../results/svm/acc_graph.txt','w')
file.write(str(test_acc))

print('Desicion')
sc_X = StandardScaler()
x_train = sc_X.fit_transform(ravel_data(x_train))
x_test = sc_X.transform(ravel_data(x_test))

classifier = DecisionTreeClassifier()
classifier = classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
test_acc = metrics.cohen_kappa_score(y_test, y_pred)

print('Accuracy Score:', metrics.accuracy_score(y_test,y_pred))
logging.info(f"Test accuracy: {test_acc}")

file = open('../results/Decision_tree/acc_graph.txt','w')
file.write(str(test_acc))

