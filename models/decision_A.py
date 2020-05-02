import logging
from utils.images import load_images
from preprocessors.baseline import ravel_data
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

kanjis = load_images('C:\\Users\\HealthTeam\\Desktop\\kanji\\kkanji\\kkanji2',
                     minimum_count=5, random_seed=0, category_limit=None)
x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)

sc_X = StandardScaler()
x_train = sc_X.fit_transform(ravel_data(x_train))
x_test = sc_X.transform(ravel_data(x_test))

classifier = DecisionTreeClassifier()
classifier = classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
test_acc = metrics.cohen_kappa_score(y_test, y_pred)

print('Accuracy Score:', metrics.accuracy_score(y_test,y_pred))
logging.info(f"Test accuracy: {test_acc}")

file = open('../result/Decision_tree/acc.txt','w')
file.write(str(test_acc))

