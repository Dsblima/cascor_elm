import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

def add_1(arr):
    return np.append(arr,1)
def addBias(matrix):#this function adds the value 1, at the last position in each array from matrix
    return np.array(list(map(add_1, matrix)))
    