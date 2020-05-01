from sklearn.naive_bayes import GaussianNB
from utils.images import load_images
from sklearn import metrics
import logging
from preprocessors.baseline import ravel_data



kanjis = load_images('C:\\Users\\HealthTeam\\Desktop\\kanji\\kkanji\\kkanji2',
                     minimum_count=5, random_seed=0, category_limit=None)
x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)

gnb = GaussianNB()
gnb.fit(ravel_data(x_train), y_train)

pred = gnb.predict(ravel_data(x_test))

test_acc = metrics.cohen_kappa_score(y_test, pred)

print('Accuracy Score:', metrics.accuracy_score(y_test,pred))
logging.info(f"Test accuracy: {test_acc}")

file = open('../results/Naive_Bayes/acc.txt','w')
file.write(str(test_acc))
