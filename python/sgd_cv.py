# coding = utf-8
""" 
Property Inspection Prediction @ Kaggle
author : littlekid
email : muyunlei@gmail.com
"""
import pandas as pd
import numpy as np
import util
from sklearn import preprocessing
import xgboost as xgb
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.svm import SVR

T1_VX = ['T1_V1','T1_V2','T1_V3','T1_V4','T1_V5','T1_V6','T1_V7','T1_V8','T1_V9',\
		'T1_V10','T1_V11','T1_V12','T1_V13','T1_V14','T1_V15','T1_V16','T1_V17']

T2_VX = ['T2_V1','T2_V2','T2_V3','T2_V4','T2_V5','T2_V6','T2_V7','T2_V8','T2_V9',\
		'T2_V10','T2_V11','T2_V12','T2_V13','T2_V14','T2_V15']

#load train and test 
train  = pd.read_csv('../train.csv', index_col=0)
test  = pd.read_csv('../test.csv', index_col=0)

# train = train[T2_VX+['Hazard']]
# test = test[T2_VX]

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

# train.drop('T2_V10', axis=1, inplace=True)
# train.drop('T2_V7', axis=1, inplace=True)
# train.drop('T1_V13', axis=1, inplace=True)
# train.drop('T1_V10', axis=1, inplace=True)

# test.drop('T2_V10', axis=1, inplace=True)
# test.drop('T2_V7', axis=1, inplace=True)
# test.drop('T1_V13', axis=1, inplace=True)
# test.drop('T1_V10', axis=1, inplace=True)


columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)

# label encode the categorical variables
inx_of_cat = dict()
for i in range(train.shape[1]):
	if type(train[1,i]) is str:
		inx_of_cat[i] = set(train[:,i])
		lbl = preprocessing.LabelEncoder()
		lbl.fit(list(train[:,i]) + list(test[:,i]))
		train[:,i] = lbl.transform(train[:,i])
		test[:,i] = lbl.transform(test[:,i])

train = train.astype(np.int64)
test = test.astype(np.int64)

ohc = preprocessing.OneHotEncoder()
train = ohc.fit_transform(train)
test = ohc.transform(test)
y = labels.values.astype(float)

scaler = preprocessing.StandardScaler().fit(train)
offtrain = scaler.transform(train[5000:,:])
offval = scaler.transform(train[:5000,:])
y = labels.values.astype(float)
offtrain_label = y[5000:]
offval_label = y[:5000]

ohc = preprocessing.OneHotEncoder()
offtrain = ohc.fit_transform(train[5000:,:])
offval = ohc.transform(train[:5000,:])
y = labels.values.astype(float)
offtrain_label = y[5000:]
offval_label = y[:5000]

model = SVR(C=20)


model = RandomForestRegressor(n_estimators=150,n_jobs=3,max_depth=7)

model = SGDRegressor(penalty='l2')


model.fit(offtrain,offtrain_label)
offtrain_pred = model.predict(offtrain)
offval_pred = model.predict(offval)
util.Gini(offval_pred,offval_label)

offpred = np.array(offtrain_pred.tolist() + offval_pred.tolist())


sgd = SGDRegressor(loss='epsilon_insensitive',penalty='elasticnet')
svm = SVR()

param_grid = {'alpha':[0.00005,0.0001,0.0005],
			'epsilon':[0.05,0.1,0.15]}

param_grid = {'C':[5,10,15,20]}

gini_scorer = metrics.make_scorer(util.Gini, greater_is_better = True)

model = grid_search.GridSearchCV(estimator = sgd, param_grid=param_grid, scoring=gini_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

model.fit(train,y)

print("Best score: %0.3f" % model.best_score_)
a_best_s = model.best_score_
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
	print("\t%s: %r" % (param_name, best_parameters[param_name]))

best_model = model.best_estimator_
best_model.fit(train,y)

ypred = best_model.predict(test)

preds = pd.DataFrame({"Id": test_ind, "Hazard": ypred})
preds = preds.set_index('Id')
preds.to_csv('../submission/pair_1000_sgd_780f_lrC2a5t3.csv')
