import numpy as np
from activation_functions import *
class Node(object):
  def __init__(self):
    self.weightsArray =[]
    self.input = 0.2
    self.output = []
    self.sensibity = 0
  
  def _multiplyInputByWeights(self):
    self.output = self.input*self.weightsArray
    # print(self.output)
    # return self.output
  
  def _setOutput(self,value):
    self.output = value
    
  def _getInput(self):
    return self.input
    
  def _setInput(self,value):
    self.input = value
  
  def _init_weights(self,numNodesNextLayer = 1):
    self.weightsArray = np.random.rand(1,numNodesNextLayer)