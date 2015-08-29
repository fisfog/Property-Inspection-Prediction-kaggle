#-*- coding:utf-8 -*-
"""
pairwiseLR
"""

import util
import numpy as np
import scipy as sp
import sys
import time

def sigmoid(x):
	r = x
	if x > 19:
		r = 19
	if x < -19:
		r = -19
	return 1.0 / (1 + np.exp(r))

def be_positive(x):
	return sigmoid(-x)

def be_negative(x):
	return 1.0-be_positive(x)

def pair_dataset(data,label):
	L = len(data)
	assert(L == len(label))
	pair_data = []
	pair_label = []
	for i in xrange(0,L-1):
		for j in xrange(i+1,L):
			pair_data.append(data[i].tolist()+data[j].tolist())
			lb = 0 if label[i]<=label[j] else 1
			pair_label.append(lb)
	return np.array(pair_data),np.array(pair_label)

def pair_dataset_generator(data,label):
	L = data.shape[0]
	assert(L == label.shape[0])
	for i in xrange(0,L-1):
		for j in xrange(i+1,L):
			pair_data = data[i].tolist()+data[j].tolist()
			lb = 0 if label[i]<=label[j] else 1
			yield np.array(pair_data),lb

def pair_dataset_sto_generator(data,label):
	L = data.shape[0]
	assert(L == label.shape[0])
	num_samples = data.shape[0] * (data.shape[0]-1) / 2
	for k in xrange(num_samples):
		i = np.random.randint(0,L-1)
		j = np.random.randint(i+1,L)
		pair_data = data[i].tolist()+data[j].tolist()
		lb = 0 if label[i]<=label[j] else 1
		yield np.array(pair_data),lb

def pair_dataset_sto_sparse_generator(data,label,sample):
	L = len(sample)
	num_samples = L * (L-1) / 2
	nf = data.shape[1]
	for k in xrange(num_samples):
		i = np.random.randint(0,L-1)
		j = np.random.randint(i+1,L)
		ainx = sample[i]
		binx = sample[j]
		alist = data.getrow(ainx).toarray().reshape(nf).tolist()
		blist = data.getrow(binx).toarray().reshape(nf).tolist()
		pair_data = alist + blist
		lb = 0 if label[ainx]<=label[binx] else 1
		yield sp.sparse.csr_matrix(pair_data),lb

def sgd_lr_4sps(data,label,sample,param):
	start = time.time()
	L = len(sample)
	num_samples = L * (L-1) / 2
	num_features = data.shape[1] * 2
	print "#Samples:%d, #Features:%d"%(num_samples,num_features)

	weights = np.random.random(num_features)
	alpha = param['alpha']
	iters = param['iters']
	gamma = param['gamma']

	for it in xrange(iters):
		print "iter%d "%it
		ic = 0
		for pi,py in pair_dataset_sto_sparse_generator(data,label,sample):
			output = sigmoid(np.sum(weights[pi.indices]))
			err = output - py
			weights = weights - alpha * (pi.toarray().reshape(num_features) * err + gamma * weights)

			util.view_bar(ic,num_samples)
			ic += 1

	print "train complete! took %.2fs"%(time.time()-start)
	return weights

def sgd_lr(data,label,param):
	start = time.time()

	num_samples = data.shape[0] * (data.shape[0]-1) / 2
	num_features = data.shape[1] * 2

	print "#Samples:%d, #Features:%d"%(num_samples,num_features)

	weights = np.random.random(num_features)
	alpha = param['alpha']
	iters = param['iters']
	gamma = param['gamma']

	for it in xrange(iters):
		print "iter%d "%it
		ic = 0
		for pi,py in pair_dataset_sto_generator(data,label):
			output = sigmoid(np.dot(pi, weights))
			err = output - py
			weights = weights - alpha * (pi * err + gamma * weights)

			# sys.stdout.write(str(ic*100.0/num_samples)+'%\r')
			util.view_bar(ic,num_samples)
			ic += 1

	print "train complete! took %.2fs"%(time.time()-start)
	return weights

def pwlr_predict(train_data,train_label,test_data,weights):
	start = time.time()
	print "Prediction Begin!"

	L = len(train_data)
	result = np.zeros(len(test_data))
	num_test = len(test_data)
	ic = 0
	for i in xrange(len(test_data)):
		test_instance = test_data[i]
		for j in xrange(L):
			train_instance = train_data[j]
			sample_instance = test_instance + train_instance
			result[i] += (1+be_positive(np.dot(sample_instance,weights))) * train_label[j]

			# sys.stdout.write(str((i+1)*(j+1)*100.0/(num_test*L))+'%\r')
			util.view_bar(ic,num_test*L)
			ic += 1

	print "prediction complete! took %.2fs"%(time.time()-start)
	return result/L

def pwlr_predict4sps_online(train_data,train_label,sample,test_data,weights):
	start = time.time()
	print "Prediction Begin!"

	L = len(sample)
	result = np.zeros(test_data.shape[0])
	num_test = test_data.shape[0]
	ic = 0
	for i in xrange(num_test):
		test_instance = test_data.getrow(i).toarray().tolist()
		for j in xrange(L):
			train_instance = train_data.getrow(sample[j]).toarray().tolist()
			sample_instance = sp.sparse.csr_matrix(test_instance + train_instance)
			result[i] += (1+be_positive(np.sum(weights[sample_instance.indices]))) * train_label[sample[j]]

			# sys.stdout.write(str((i+1)*(j+1)*100.0/(num_test*L))+'%\r')
			util.view_bar(ic,num_test*L)
			ic += 1

	print "prediction complete! took %.2fs"%(time.time()-start)
	return result/L

def pwlr_predict4sps_offline(train_data,train_label,sample,weights):
	start = time.time()
	print "Prediction Begin!"

	L = len(sample)
	test_inx = list(set([i for i in xrange(train_data.shape[0])])-set(sample))
	result = np.zeros(len(test_inx))
	num_test = len(test_inx)
	ic = 0
	for i in xrange(num_test):
		test_instance = train_data.getrow(test_inx[i]).toarray().tolist()
		for j in xrange(L):
			train_instance = train_data.getrow(sample[j]).toarray().tolist()
			sample_instance = sp.sparse.csr_matrix(test_instance + train_instance)
			result[i] += (1+be_positive(np.sum(weights[sample_instance.indices]))) * train_label[sample[j]]

			# sys.stdout.write(str((i+1)*(j+1)*100.0/(num_test*L))+'%\r')
			util.view_bar(ic,num_test*L)
			ic += 1

	print "prediction complete! took %.2fs"%(time.time()-start)
	return result/L


