from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from utils.images import load_images
from preprocessors.baseline import ravel_data
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

kanjis = load_images('C:\\Users\\takac\\OneDrive\\Desktop\\asd',
                     minimum_count=5, random_seed=0, category_limit=None)
x_train, y_train, x_test, y_test = kanjis.train_test_split(0.6)

sc_X = StandardScaler()
x_train = sc_X.fit_transform(ravel_data(x_train))
x_test = sc_X.transform(ravel_data(x_test))

classifier = DecisionTreeClassifier()
classifier = classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

print('Accuracy Score:', metrics.accuracy_score(y_test,y_pred))
