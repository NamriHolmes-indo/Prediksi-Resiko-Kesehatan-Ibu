import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import *
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree  import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('Maternal Health Risk Data Set.csv')

df.info()

df.head(5)

df['RiskLevel'].value_counts()

resiko = {'low risk':0, 'mid risk':1, 'high risk':2}
df['RiskLevel'] = df['RiskLevel'].map(resiko)

x = df.drop(labels=['RiskLevel'], axis=1)
y = df['RiskLevel']

df['RiskLevel'].value_counts()

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.95, random_state=1)

clf = AdaBoostClassifier(random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
adaBoostReport = explained_variance_score(y_pred, y_test)

rand_regr = RandomForestRegressor(n_estimators=400,random_state=0)
rand_regr.fit(X_train, y_train)
random=rand_regr.score(X_test,y_test)
predictions = rand_regr.predict(X_test)
RandForReport = explained_variance_score(predictions,y_test)

est=GradientBoostingRegressor(n_estimators=400, max_depth=5, loss='ls',min_samples_split=2,learning_rate=0.1).fit(X_train, y_train)
gradient=est.score(X_test,y_test)

pred = est.predict(X_test)
GradBoostReport = explained_variance_score(pred,y_test)

ada=AdaBoostRegressor(n_estimators=50, learning_rate=0.2,loss='exponential').fit(X_train, y_train)
pred=ada.predict(X_test)
adab=ada.score(X_test,y_test)
predict = ada.predict(X_test)
aadReport = explained_variance_score(predict,y_test)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
  max_depth = None,
  min_samples_split = 2
)

dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

dtreport = classification_report(y_test, y_pred)
print(dtreport)

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3), random_state=1)
mlp.fit(X_train,y_train)
mlp_sc = mlp.score(X_test,y_test)
mlppredict = mlp.predict(X_test)
exp_mlp = classification_report(mlppredict,y_test)

annReport = explained_variance_score(mlppredict,y_test)

knn = KNeighborsClassifier(n_neighbors=49,weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=None)
knn.fit(X_train,y_train)
knn_sc = knn.score(X_test,y_test)
knnpredict = knn.predict(X_test)
exp_knn = explained_variance_score(knnpredict,y_test)

knnReport = classification_report(knnpredict,y_test)

print(knnReport)
print(exp_mlp)

#untuk Save Model Machine Learning
rand_regr = RandomForestRegressor(n_estimators=400,random_state=0)
rand_regr.fit(x, y)
import pickle
with open('RandomForest.pkl', 'wb') as files:
  pickle.dump(rand_regr, files)

dt = dt.fit(x, y)
import pickle
with open('DecissionTree.pkl', 'wb') as files:
  pickle.dump(dt, files)

for_test = []
for_test.append(int(25))
for_test.append(int(100))
for_test.append(int(80))
for_test.append(int(89))
for_test.append(int(86))
for_test.append(int(56))
predictions = int(rand_regr.predict([for_test]))
DT = int(dt.predict([for_test]))
if predictions == 1:
  print("Prediksi Dengan Random Regressor : Resiko Sedang")
elif predictions ==2:
  print("Prediksi Dengan Random Regressor : Resiko Tinngi")
else:
  print("Prediksi Dengan Random Regressor : Resiko Rendah")

if dt == 1:
  print("Prediksi Dengan Decision Tree : Resiko Sedang")
elif dt ==2:
  print("Prediksi Dengan Decision Tree : Resiko Tinngi")
else:
  print("Prediksi Dengan Decision Tree : Resiko Rendah")