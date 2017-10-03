from openpyxl import load_workbook
import numpy



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

# We do the same for string values in the feature list (dictvectorizer ignores non-string values)
from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
features = v.fit_transform(features)

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(features, labels)

# Train and predict with GuassianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
clf.predict(X_test)
print(clf.score(X_test, y_test))
