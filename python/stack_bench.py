# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 23:50:55 2015

@author: Justfor <justformaus@gmx.de>

Purpose: This script tries to implement a technique called stacking/stacked generalization.
I made this a runnable script available because I found that there isn't really any
readable code that demonstrates this technique. 
======================================================================================================
Summary:

Just to test an implementation of stacking/stacked generalization.
Using a cross-validated ExtraTrees Regressor, Random forest and Gradient Boosting Regressor.
Leaderboard: 0.361412
Improvements should be very easy.

This code is heavily inspired from the classification code shared by Emanuele (https://github.com/emanuele)
and by Eric Chio "log0" <im.ckieric@gmail.com>,  but I have made it for regression, cleaned it up to
 make it available for easy download and execution.

======================================================================================================
Methodology:

Three classifiers (ExtraTreesRegressor, RandomForestRegressor and a GradientBoostingRegressor)
are built to be stacked by a RidgeCVRegressor in the end.

Some terminologies first, since everyone has their own, I'll define mine to be clear:
- DEV SET, this is to be split into the training and validation data. It will be cross-validated.
- TEST SET, this is the unseen data to validate the generalization error of our final classifier. This
set will never be used to train.
When DEVELOPMENT is set True, then cross validation and evaluation takes place.
Otherwise (DEVELOPMENT=False) a submission file is generated.

======================================================================================================
Data Set Information:
Kaggle Competition LMP Property Inspection
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR,LinearSVR,NuSVR

import xgboost as xgb
from sklearn.linear_model import RidgeCV
from datetime import datetime

# Source of good version: https://www.kaggle.com/c/ClaimPredictionChallenge/forums/t/703/code-to-calculate-normalizedgini    
def gini(actual, pred, cmpcol = 0, sortcol = 1):
     assert( len(actual) == len(pred) )
     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
     totalLosses = all[:,0].sum()
     giniSum = all[:,0].cumsum().sum() / totalLosses
     giniSum -= (len(actual) + 1) / 2.
     return giniSum / len(actual)
 
def normalized_gini(a, p):
     return gini(a, p) / gini(a, a)


def prepare_data1():
    # load train data
    train    = pd.read_csv('../train.csv')
    test     = pd.read_csv('../test.csv')
    
    labels   = train.Hazard   
    test_ind = test.ix[:,'Id']
    train.drop('Hazard', axis=1, inplace=True)
    train_ind = train.ix[:,'Id']
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    Dropcols = ['T2_V10', 'T2_V7','T1_V13', 'T1_V10']
    train.drop( Dropcols, axis = 1, inplace=True )
    test.drop( Dropcols, axis = 1, inplace=True )
    
    catCols=['T1_V4', 'T1_V5','T1_V6', 'T1_V7', 'T1_V8', 'T1_V9', 'T1_V11', 'T1_V12', 'T1_V15',
     'T1_V16', 'T1_V17', 'T2_V3', 'T2_V5', 'T2_V11', 'T2_V12', 'T2_V13']
  
    
    trainNumX=train.drop(catCols, axis=1)
    scaler = preprocessing.StandardScaler().fit(trainNumX)
    trainNumX = scaler.transform(trainNumX)
    trainCatVecX = pd.get_dummies(train[catCols])   
    trainX = np.hstack((trainCatVecX,trainNumX))

    testNumX=test.drop(catCols, axis=1)
    scaler = preprocessing.StandardScaler().fit(testNumX)
    testNumX = scaler.transform(testNumX)
    testCatVecX = pd.get_dummies(test[catCols])  
    testX = np.hstack((testCatVecX,testNumX))
 
    return trainX, labels, train_ind, testX, test_ind

def prepare_data2():
    # load train data
    train    = pd.read_csv('../train.csv')
    test     = pd.read_csv('../test.csv')
    
    labels   = train.Hazard   
    test_ind = test.ix[:,'Id']
    train.drop('Hazard', axis=1, inplace=True)
    train_ind = train.ix[:,'Id']
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    Dropcols = ['T2_V10', 'T2_V7','T1_V13', 'T1_V10']
    train.drop( Dropcols, axis = 1, inplace=True )
    test.drop( Dropcols, axis = 1, inplace=True )
    
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

    scaler = preprocessing.StandardScaler().fit(train)
    trainX = scaler.transform(train)
    testX = scaler.transform(test)

    return trainX, labels, train_ind, testX, test_ind

def prepare_data3():
    # load train data
    # polynomial features
    train    = pd.read_csv('../train.csv')
    test     = pd.read_csv('../test.csv')

    labels   = train.Hazard   
    test_ind = test.ix[:,'Id']
    train.drop('Hazard', axis=1, inplace=True)
    train_ind = train.ix[:,'Id']
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    Dropcols = ['T2_V10', 'T2_V7','T1_V13', 'T1_V10']
    train.drop( Dropcols, axis = 1, inplace=True )
    test.drop( Dropcols, axis = 1, inplace=True )

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

    poly = preprocessing.PolynomialFeatures(degree=2)
    trainX = poly.fit_transform(train)
    testX = poly.transform(test)

    scaler = preprocessing.StandardScaler().fit(trainX)
    trainX = scaler.transform(trainX)
    testX = scaler.transform(trainX)

    return trainX, labels, train_ind, testX, test_ind

if __name__ == '__main__':    
    DEVELOP = False

    SEED = 42
    np.random.seed(SEED)

    X, Y, idx, testX, testidx = prepare_data2()
    print X.shape

    print ("Preparing models.")

    if (DEVELOP==True):
        # The DEV SET will be used for all training and validation purposes
        # The TEST SET will never be used for training, it is the unseen set.
        dev_cutoff = int(round(len(Y) * 4/5))
        X_dev = X[:dev_cutoff]
        Y_dev = Y[:dev_cutoff]
        X_test = X[dev_cutoff:]
        X_test_xgbMat = xgb.DMatrix(X_test)
        Y_test = Y[dev_cutoff:]            
    else:    # else submit    
        X_dev = X
        Y_dev = Y
        X_test = testX
        X_test_xgbMat = xgb.DMatrix(X_test)

    n_trees = 800
    n_folds = 5
    # penalty='l2'
    # XGB PARAMS
    xgb_params = {}
    # xgb_params["booster"] = "gblinear"
    xgb_params["objective"] = "reg:linear"
    xgb_params["eta"] = 0.005
    xgb_params["min_child_weight"] = 50
    xgb_params["subsample"] = 0.7
    xgb_params["scale_pos_weight"] = 1.0
    xgb_params["silent"] = 1
    xgb_params["max_depth"] = 9
    num_rounds = 5000

    # Our level 0 classifiers
    clfs = [
        # SGDRegressor(penalty='l1'),
        xgb,
        ExtraTreesRegressor(n_estimators = 900,n_jobs=-1),
        RandomForestRegressor(n_estimators = 900,n_jobs=-1),
        GradientBoostingRegressor(n_estimators = 450),
        # SVR(C=5)
    ]

    # Ready for cross validation
    skf = KFold(n=X_dev.shape[0], n_folds=n_folds)

    # Pre-allocate the data
    blend_train = np.zeros((X_dev.shape[0], len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((X_test.shape[0], len(clfs))) # Number of testing data x Number of classifiers

    print ("Calculating pre-blending values.")
    start_time = datetime.now()

    cv_results = np.zeros((len(clfs), len(skf)))  # Number of classifiers x Number of folds

    # For each classifier, we train the number of fold times (=len(skf))
    for j, clf in enumerate(clfs):
        print ('\nTraining classifier [%s]%s' % (j, clf))
        blend_test_j = np.zeros((X_test.shape[0], len(skf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(skf):
            #print ('Fold [%s]' % (i))
            
            # This is the training and validation set
            X_train = X_dev[train_index]
            Y_train = Y_dev[train_index]
            X_cv = X_dev[cv_index]
            Y_cv = Y_dev[cv_index]


            #print("fit")
            if clf==xgb:
                plst = list(xgb_params.items())
                xgtrain = xgb.DMatrix(X_train,label=Y_train)
                xgval = xgb.DMatrix(X_cv,label=Y_cv)
                watchlist = [(xgtrain, 'train'),(xgval, 'val')]
                bst = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=5)
                one_result = bst.predict(xgval)
                blend_test_j[:, i] = bst.predict(X_test_xgbMat) 
            else:
                clf.fit(X_train, Y_train)
                one_result = clf.predict(X_cv)
                blend_test_j[:, i] = clf.predict(X_test)
            #print("blend")
            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            blend_train[cv_index, j] = one_result
            score = normalized_gini(Y_cv, blend_train[cv_index, j])
            cv_results[j,i] = score
            score_mse = metrics.mean_absolute_error(Y_cv, one_result)    
            print ('Fold [%s] norm. Gini = %0.5f, MSE = %0.5f' % (i, score, score_mse))     
        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)      
        print ('Clf_%d Mean norm. Gini = %0.5f (%0.5f)' % (j, cv_results[j,].mean(), cv_results[j,].std()))

    end_time = datetime.now()
    time_taken = end_time - start_time
    print ("Time taken for pre-blending calculations: ", time_taken)

    print ("CV-Results", cv_results)
    
    # Start blending!    
    print ("Blending models.")

    alphas = [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    bclf = RidgeCV(alphas=alphas, normalize=True, cv=5)
    bclf.fit(blend_train, Y_dev)       
    print ("Ridge Best alpha = ", bclf.alpha_)
   
    # Predict now
    Y_test_predict = bclf.predict(blend_test)
    
    if (DEVELOP):
            score1 = metrics.mean_absolute_error(Y_test, Y_test_predict)
            score = normalized_gini(Y_test, Y_test_predict)
            print ('Ridge MSE = %s normalized Gini = %s' % (score1, score))
    else: # Submit! and generate solution
        score = cv_results.mean()      
        print ('Avg. CV-Score = %s' % (score))
        #generate solution
        submission = pd.DataFrame({"Id": testidx, "Hazard": Y_test_predict})
        submission = submission.set_index('Id')
        submission.to_csv("../submission/stacking_xgb_rf_skGBRT_SVR_data1.csv") 