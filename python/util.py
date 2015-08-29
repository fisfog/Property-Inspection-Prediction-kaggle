import numpy as np
import pandas as pd
import os,sys,string
import time

def gini(solution, submission):
    df = zip(solution, submission)
    df = sorted(df, key=lambda x: (x[1],x[0]), reverse=True)
    rand = [float(i+1)/float(len(df)) for i in range(len(df))]
    totalPos = float(sum([x[0] for x in df]))
    cumPosFound = [df[0][0]]
    for i in range(1,len(df)):
        cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
    Lorentz = [float(x)/totalPos for x in cumPosFound]
    Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
    return sum(Gini)

def normalized_gini(solution, submission):
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

# Simple implementation of the (normalized) gini score in numpy
# Fully vectorized, no python loops, zips, etc.
# Significantly (>30x) faster than previous implementions
def Gini(y_pred, y_true):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true



def stacking(predlist,outfile):
	pred_df = []
	for f in predlist:
		pred_df.append(pd.read_csv(f))

	all_df = pred_df[0]
	for i in xrange(1,len(pred_df)):
		all_df = pd.merge(all_df,pred_df[i],on='Id')
	data = all_df.drop(['Id'],axis=1).values
	idx = all_df.Id.values
	print idx
	pred = []
	for i in xrange(len(all_df)):
		pred.append(data[i].sum()/len(pred_df))
	submission = pd.DataFrame({"Id": idx, "Hazard": pred})
	submission.to_csv(outfile, columns = ["Id","Hazard"],index=False)


def view_bar(n, s):
	rate = float(n) / float(s)
	rate_num = rate * 100
	print '\r%.3f%% :' %(rate_num),
	sys.stdout.flush()
