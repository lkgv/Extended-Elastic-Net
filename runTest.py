# -*- coding: utf-8 -*-
"""
Created on Thu Feb 08 19:31:14 2018

@author: kavinche
"""

import scipy as sc
import pandas as pd
import time
import numpy as np
import math
from scipy import stats
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LassoLarsIC, ElasticNet, ElasticNetCV, Ridge, RidgeCV

np.random.seed(1)

X = np.loadtxt('C:/Users/kavinche/Desktop/Exp-assoc/new_data_raw.csv', dtype = 'int', delimiter = ",", skiprows = 1)
#b = a[:,1:426].astype('int')

'''
beita = [0]*1999
rd1 =1 - 2*np.random.rand(1,18)
rd2 =1 - 2*np.random.rand(1,12)
beita1 = beita + rd1[0].tolist() + [0]*991 + rd2[0].tolist() + [0]*2095
beita1 = np.array(beita1)
beita1 = np.expand_dims(beita1, axis=1)
'''
beta = np.loadtxt('C:/Users/kavinche/Desktop/Exp-assoc/beta.txt')
beta = np.expand_dims(beta, axis = 1)

e = np.random.normal(loc=0, scale=1, size=X.shape[0])
e = np.expand_dims(e, axis = 1)

y = np.dot(X, beta) + e
#y = np.squeeze(y)

#y = y.tolist()


print "(1) it's trained by lr!"
t = time.time()
lr = LinearRegression()
lr.fit(X,y)
t_use = time.time() - t
params = np.append(lr.intercept_,lr.coef_)
#predictions = lr.predict(x)
print "it takes %f s" % t_use
np.savetxt("lr.coef_.txt", lr.coef_)



print "(2) it's trained by lasso!"
t = time.time()
model_lasso = Lasso(alpha = 0.5).fit(X,y)
params_lasso = np.append(model_lasso.intercept_, model_lasso.coef_)
t_use = time.time() - t
print "it takes %f s" % t_use
np.savetxt("model_lasso.coef_.txt", model_lasso.coef_)



print "(3) it's trained by lassoCV!"
t = time.time()
model_lassoCV = LassoCV(cv=10).fit(X,y)
params_lassoCV = np.append(model_lassoCV.intercept_, model_lassoCV.coef_)
t_use = time.time() - t
print "it takes %f s" % t_use
np.savetxt("model_lassoCV.coef_.txt", model_lassoCV.coef_)


print "(4) it's trained by lassoLarsBIc!"
t = time.time()
model_bic = LassoLarsIC(criterion='bic').fit(X,y)
params_bic = np.append(model_bic.intercept_, model_bic.coef_)
t_use = time.time() - t
print "it takes %f s" % t_use
np.savetxt("model_bic.coef_.txt", model_bic.coef_)


print "(5) it's trained by lassoLarsAIc!"
t = time.time()
model_aic = LassoLarsIC(criterion='aic').fit(X,y)
params_aic = np.append(model_aic.intercept_, model_aic.coef_)
t_use = time.time() - t
print "it takes %f s" % t_use
np.savetxt("model_aic.coef_.txt", model_aic.coef_)



print "(6) it's trained by elasticNet!"
t = time.time()
model_en = ElasticNet(l1_ratio = 0.9).fit(X,y)
params_en = np.append(model_en.intercept_, model_en.coef_)
t_use = time.time() - t
print "it takes %f s" % t_use
np.savetxt("model_en.coef_.txt", model_en.coef_)




print "(7) it's trained by elasticNetCV!"
t = time.time()
model_enCv = ElasticNetCV(l1_ratio = 0.5, cv = 10).fit(X,y)
params_enCv = np.append(model_enCv.intercept_, model_enCv.coef_)
t_use = time.time() - t
print "it takes %f s" % t_use
np.savetxt(" model_enCv.coef_.txt",  model_enCv.coef_)



print "(8) it's trained by Ridge!"
t = time.time()
model_ridge = Ridge(alpha = 0.5).fit(X,y)
params_ridge = np.append(model_ridge.intercept_, model_ridge.coef_)
t_use = time.time() - t
print "it takes %f s" % t_use
np.savetxt("model_ridge.coef_.txt", model_ridge.coef_)



print "(9) it's trained by RidgeCV!"
t = time.time()
model_rCV = RidgeCV(alphas=(0.1, 1.0, 10.0)).fit(X,y)
params_rCV = np.append(model_rCV.intercept_, model_rCV.coef_)
t_use = time.time() - t
print "it takes %f s" % t_use
np.savetxt("model_rCV.coef_.txt", model_rCV.coef_)



print "it's done!"