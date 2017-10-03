from openpyxl import load_workbook
import numpy



d = {}
featureNames = []
wb = load_workbook(filename='attacksignatures.xlsx')
ws = wb['dataset1']

firstCell = True
for tpl in tuple(ws.columns):
    d[tpl[0].value] = []
    d[tpl[0].value].extend([col.value for col in tpl[1:]])
    featureNames.append(tpl[0].value)

print("Features: " + ", ".join(featureNames))

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

features = []
features.extend([{} for row in d["attackclass"]])

for f in featureNames[:-2]:
    rownum = 0
    for val in d[f]:
        features[rownum][f] = val
        rownum += 1

labels = d['attackclass']

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)

from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
features = v.fit_transform(features)

from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=41)
clf = GaussianNB()
results = []
for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    clf.fit(X_train, y_train)
    clf.predict(X_test)
    score = clf.score(X_test, y_test)
    results.append(tuple([X_train, X_test, y_train, y_test, score]))
print('scores:\n' + '\n'.join([str(tup[4]) for tup in results]))
