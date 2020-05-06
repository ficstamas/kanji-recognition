from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from utils.images import load_images
from sklearn import metrics
import logging
import numpy as np
from preprocessors.baseline import ravel_data
import cv2
from skimage.feature import hog


kanjis = load_images('C:\\Users\\HealthTeam\\Desktop\\kanji\\kkanji\\kkanji2',
                     minimum_count=5, random_seed=0, category_limit=None)
x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)
hog_train_x = []
hog_test_x = []
for img in x_train:
    fd, hog_image = hog(img, orientations = 8, pixels_per_cell = (3, 3),
                            cells_per_block = (1,1), visualize = True, multichannel = False)

    hog_train_x.append(hog_image)


hog_train_x = np.array(hog_train_x)
print('x_train kesz, x_test jön')

for img in x_test:
    fd, hog_image = hog(img, orientations = 8, pixels_per_cell = (3, 3),
                            cells_per_block = (1,1), visualize = True, multichannel = False)
    hog_test_x.append(hog_image)
hog_test_x = np.array(hog_test_x)

print('x_test kész, tanítás jön')
#print(hog_train_x.shape,x_train.shape)
'''
gnb = GaussianNB()
gnb.fit(ravel_data(hog_train_x), y_train)

pred = gnb.predict(ravel_data(hog_test_x))

test_acc = metrics.cohen_kappa_score(y_test, pred)

print('Accuracy Score:', metrics.accuracy_score(y_test,pred))
logging.info(f"Test accuracy: {test_acc}")

file = open('../results/Naive_Bayes/acc_hog.txt','w')
file.write(str(test_acc))
'''
print('svm')
model = LinearSVC()
model.fit(ravel_data(hog_train_x), y_train)

predictions = model.predict(ravel_data(hog_test_x))
test_acc = metrics.cohen_kappa_score(y_test, predictions)

print('Accuracy Score:', metrics.accuracy_score(y_test,predictions))
logging.info(f"Test accuracy: {test_acc}")

file = open('../results/svm/acc_hog.txt','w')
file.write(str(test_acc))

print('tree')

sc_X = StandardScaler()
x_train = sc_X.fit_transform(ravel_data(hog_train_x))
x_test = sc_X.transform(ravel_data(hog_test_x))

classifier = DecisionTreeClassifier()
classifier = classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
test_acc = metrics.cohen_kappa_score(y_test, y_pred)

print('Accuracy Score:', metrics.accuracy_score(y_test,y_pred))
logging.info(f"Test accuracy: {test_acc}")

file = open('../results/Decision_tree/acc_hog.txt','w')
file.write(str(test_acc))
