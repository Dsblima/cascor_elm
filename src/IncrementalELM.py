import numpy as np
import pandas as pd

def qrFatorization(m,y):
    gamma, delta = np.linalg.qr(m)
    np.fill_diagonal(delta, 1)
    prod = np.dot(np.linalg.pinv(delta), np.linalg.pinv(gamma))
    # prod = np.dot(delta, gamma.T)
    # prod = np.dot( q, r)
    return np.dot(prod,y)

# qr fatorization in the input matrix 
# calculate output weights
# initialize input weights
# freeze inout weights
# evaluate each candidate
# add to the network the candidate with maximum reduction
# Update matrix