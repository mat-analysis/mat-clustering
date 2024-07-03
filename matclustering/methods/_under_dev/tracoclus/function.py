import pandas as pd
import numpy as np

def cost_function(numOfObj, numOfAtt, cov=0, noise=0, verbose=True):
    if verbose:
        print('Num. objs: {0:2d}, Num. att: {1:2d}, Num. covered: {2:2d}, Num. noise: {3:2d}'.format(numOfObj,numOfAtt,cov,noise))
    return ((numOfObj+numOfAtt) - (numOfObj*numOfAtt)) + cov + noise