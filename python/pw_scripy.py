import util
import os
import numpy as np
import pairwise
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from sklearn import preprocessing



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

scaler = preprocessing.StandardScaler()

train = scaler.fit_transform(train)
test = scaler.transform(test)

param = {}
param['alpha'] = 1e-5
param['gamma'] = 2
param['iters'] = 2

sample = 500

offset_val = []
while len(set(offset_val)) < sample:
	offset_val.append(np.random.randint(train.shape[0]))

offset_val = list(set(offset_val))
offset_test = list(set([i for i in xrange(len(train))])-set(offset_val))

train_data = train[offset_val]
train_label = y[offset_val]

offline_test = train[offset_test]
offline_test_label = y[offset_test]

# offline validation
weights = pairwise.sgd_lr(train_data,train_label,param)
test_score = pairwise.pwlr_predict(train_data.tolist(),train_label.tolist(),offline_test.tolist(),weights)
print "Offline Val:%.4f"%(util.normalized_gini(test_score,offline_test_label))

# CV
kf = KFold(train_data.shape[0],n_folds=3)
cv_result = []
k = 0

for train_idx,test_idx in kf:
	cv_train_data = train_data[train_idx]
	cv_train_label = train_label[train_idx]
	cv_test_data = train_data[test_idx]
	cv_test_label = train_label[test_idx]
	weights = pairwise.sgd_lr(cv_train_data,cv_train_label,param)
	test_score = pairwise.pwlr_predict(cv_train_data.tolist(),cv_train_label.tolist(),cv_test_data.tolist(),weights)
	cv_result.append(util.normalized_gini(test_score,cv_test_label))
	print "CV_%d: %.4f"%(k,util.normalized_gini(test_score,cv_test_label))
	k += 1

print "CV Mean:%.4f, Std:%.4f"%(np.array(cv_result).mean(),np.array(cv_result).std())

