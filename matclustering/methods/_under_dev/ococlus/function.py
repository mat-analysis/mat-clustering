import pandas as pd
import numpy as np

def cost_function(numOfObj, numOfAtt, cov=0,noise=0):
    if VERBOSE:
        print("Num. objs: {0:2d}, Num. att: {1:2d}, Num. noise: {2:2d}".format(numOfObj,numOfAtt,noise))
    return ((numOfObj+numOfAtt) - (numOfObj*numOfAtt)) + cov + (2*noise)