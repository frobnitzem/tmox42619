#!/usr/bin/python3

import numpy as np
import h5py
from scipy import tanh
import math

def tanhInt(x,bits):
    y = 2**(bits-1)*(1+tanh(x.astype(float)))
    return y.astype(type(x[0]))
def pkey(p):
    return 'port_%i'%p

def mypoly(x,order=4):
	result = np.ones((x.shape[0],order+1),dtype=float)
	result[:,1] = x.copy()
	if order < 2:
		return result
	for p in range(2,order+1):
		result[:,p] = np.power(result[:,1],int(p))
	return result

def fitpoly(x,y,order=4):
    assert len(x)==len(y)
    assert len(x)>order
    x0 = np.mean(np.array(x))
    theta = np.linalg.pinv( mypoly(np.array(x-x0).astype(float),order=order) ).dot(np.array(y).astype(float)) # fit a polynomial (order 3) to the points
    return x0,theta

def getcentroid(inds,spec):
    x = inds
    y = spec 
    num = np.sum(x*y)
    denom = np.sum(y)
    if (denom>0):
        return int(num/denom)
    return 0

def makehist(fname,bins=(np.arange(2**10+1)-2**9)*2**10):
    with h5py.File(fname,'r') as f:
        for p in f.keys():
            data = []
            for k in f[p]['waves'].keys():
                data += list(f['port_0']['waves'][k][()])
            h,b = np.histogram(data,bins)
            np.savetxt('histsig.%s.dat'%p,np.column_stack((b[:-1],h)),fmt='%i')
    return

