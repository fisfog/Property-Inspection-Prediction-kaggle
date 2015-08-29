# coding = utf-8
""" 
Property Inspection Prediction @ Kaggle
author : littlekid
email : muyunlei@gmail.com
"""
import pandas as pd
import numpy as np
import scipy as sp
import util
from sklearn import preprocessing
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression

#load train and test 
train  = pd.read_csv('../train.csv', index_col=0)
test  = pd.read_csv('../test.csv', index_col=0)

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

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

train = train.astype(float)
test = test.astype(float)

params = {}
params["objective"] = "reg:linear"
# params["objective"] = "rank:pairwise"

params["eta"] = 0.01
params["min_child_weight"] = 20
params["subsample"] = 0.75
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 5

plst = list(params.items())


# Val
offset_val = []
while len(set(offset_val)) < 5000:
	offset_val.append(np.random.randint(len(train)))

offset_val = list(set(offset_val))
offset_train = list(set([i for i in xrange(len(train))])-set(offset_val))

y = labels.values.astype(float)
offtrain = train[offset_train]
offlabel = y[offset_train]
val = train[offset_val]
vallabel = y[offset_val]

num_rounds = 2000
xgtest = xgb.DMatrix(test)

xgtrain = xgb.DMatrix(offtrain, label=offlabel)
xgval = xgb.DMatrix(val, label=vallabel)

watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=10)
offline_val_preds = model.predict(xgval)
util.Gini(offline_val_preds,vallabel)

preds1 = model.predict(xgtest)


preds = preds1 + preds2


preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('../submission/xgboost_onehot_01.csv')
