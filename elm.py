import numpy as np
from util import *

class ELM(object):
    def init_weights(self,xcol,hiddennodes):
         return np.random.rand(xcol,hiddennodes)
    def insert_bias(self,matrix):
        return addBias(matrix)
    def pinv(self,matrix):
        return np.linalg.pinv(matrix)