'''
This benchmark uses xgboost and early stopping to achieve a score of 0.38019
In the liberty mutual group: property inspection challenge

Based on Abhishek Catapillar benchmark
https://www.kaggle.com/abhishek/caterpillar-tube-pricing/beating-the-benchmark-v1-0

@author Devin

Have fun;)
'''

import pandas as pd
import numpy as np 
from sklearn import preprocessing
import xgboost as xgb
import util

#load train and test 
train  = pd.read_csv('../train.csv', index_col=0)
test  = pd.read_csv('../test.csv', index_col=0)

train = train[T2_VX+['Hazard']]
test = test[T2_VX]

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

train.drop('T2_V10', axis=1, inplace=True)
train.drop('T2_V7', axis=1, inplace=True)
# train.drop('T1_V13', axis=1, inplace=True)
# train.drop('T1_V10', axis=1, inplace=True)

test.drop('T2_V10', axis=1, inplace=True)
test.drop('T2_V7', axis=1, inplace=True)
# test.drop('T1_V13', axis=1, inplace=True)
# test.drop('T1_V10', axis=1, inplace=True)


columns = train.columns
test_ind = test.index

train = np.array(train)
test = np.array(test)

# label encode the categorical variables
for i in range(train.shape[1]):
	if type(train[1,i]) is str:
		lbl = preprocessing.LabelEncoder()
		lbl.fit(list(train[:,i]) + list(test[:,i]))
		train[:,i] = lbl.transform(train[:,i])
		test[:,i] = lbl.transform(test[:,i])

train = train.astype(float)
test = test.astype(float)

params = {}
params["booster"] = "gblinear"
params["objective"] = "reg:linear"
# params["objective"] = "multi:softmax"
params["eta"] = 0.01
params["min_child_weight"] = 20
params["subsample"] = 0.75
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 5

plst = list(params.items())

#Using 5000 rows for early stopping. 
offset = 5000

num_rounds = 2000
xgtest = xgb.DMatrix(test)

#create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

#train using early stopping and predict
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=5)
offline_val_preds = model.predict(xgval)
util.Gini(offline_val_preds,labels[:offset].values)

preds1 = model.predict(xgtest)

# reverse train and labels and use different 5k for early stopping. 
# this adds very little to the score but it is an option if you are concerned about using all the data. 
train = train[::-1,:]
labels = labels[::-1]

xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=5)
preds2 = model.predict(xgtest)

#combine predictions
#since the metric only cares about relative rank we don't need to average
preds = preds1 + preds2

#generate solution
preds = pd.DataFrame({"Id": test_ind, "Hazard": preds})
preds = preds.set_index('Id')
preds.to_csv('../submission/xgboostX2_rf_skGBRT.csv')

