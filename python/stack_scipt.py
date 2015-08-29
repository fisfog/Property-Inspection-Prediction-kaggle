DEVELOP = False

SEED = 42   
np.random.seed(SEED)

X, Y, idx, testX, testidx = stack_bench.prepare_data()

print ("Preparing models.")
  
if (DEVELOP==True):
    dev_cutoff = int(round(len(Y) * 4/5))
    X_dev = X[:dev_cutoff]
    Y_dev = Y[:dev_cutoff]
    X_test = X[dev_cutoff:]
    Y_test = Y[dev_cutoff:]
else:
    X_dev = X
    Y_dev = Y
    X_test = testX
        
n_trees = 30
n_folds = 5
  
# Our level 0 classifiers
clfs = [
    ExtraTreesRegressor(n_estimators = n_trees *2),
    RandomForestRegressor(n_estimators = n_trees),
    GradientBoostingRegressor(n_estimators = n_trees)
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
        clf.fit(X_train, Y_train)
        
        #print("blend")
        # This output will be the basis for our blended classifier to train against,
        # which is also the output of our classifiers
        one_result = clf.predict(X_cv)
        blend_train[cv_index, j] = one_result
        score = stack_bench.normalized_gini(Y_cv, blend_train[cv_index, j])
        cv_results[j,i] = score
        score_mse = metrics.mean_absolute_error(Y_cv, one_result)    
        print ('Fold [%s] norm. Gini = %0.5f, MSE = %0.5f' % (i, score, score_mse)) 
        blend_test_j[:, i] = clf.predict(X_test)       
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
        score = stack_bench.normalized_gini(Y_test, Y_test_predict)
        print ('Ridge MSE = %s normalized Gini = %s' % (score1, score))
else: # Submit! and generate solution
    score = cv_results.mean()      
    print ('Avg. CV-Score = %s' % (score))
    #generate solution
    submission = pd.DataFrame({"Id": testidx, "Hazard": Y_test_predict})
    submission = submission.set_index('Id')
    submission.to_csv("./submission/bench_gen_stacking.csv") 