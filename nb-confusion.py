from openpyxl import load_workbook
from sklearn.metrics import confusion_matrix
import numpy
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

d = {}
featureNames = []
wb = load_workbook(filename='attacksignatures.xlsx')
ws = wb['dataset1']

# Split colomns into dictionary entries with values as array
for tpl in tuple(ws.columns):
    d[tpl[0].value] = []
    d[tpl[0].value].extend([col.value for col in tpl[1:]]) # ignore the first entry (feature names) add the rest to the array
    featureNames.append(tpl[0].value) 

print("Features: " + ", ".join(featureNames))

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# feature array initialisation
# creates an array of dictionairies with the same length as the amount of datapoints
features = []
features.extend([{} for row in d["attackclass"]])

# Fill the features array with values, excluding the last 2 columns (our label, and excel formula that categorises the label)
for f in featureNames[:-2]:
    rownum = 0
    for val in d[f]:
        features[rownum][f] = val
        rownum += 1

# labels are text, but we need numbers, use LabelEncoder to encode labels to numeric categories
from sklearn import preprocessing
labels = d['attackclass']
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)
labelNames = le.classes_

# We do the same for string values in the feature list (dictvectorizer ignores non-string values)
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
features = v.fit_transform(features)

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(features, labels)

# Train and predict with GuassianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(clf.score(X_test, y_test))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
numpy.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labelNames,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labelNames, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
