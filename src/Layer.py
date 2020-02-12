from Node import *
from activation_functions import * 
import numpy as np
class Layer:
  def __init__(self, numNodes):
    self.nodes = []
    for i in range(numNodes):
      self.nodes.append(Node())
      self.total = []
      self.pred = []
   
  def _init_weights(self,numNodes):    
    for n in self.nodes:
      n._init_weights(numNodes)      
      
  def _get_output(self):#Obtain the sum of the multiplication of the node value for all weights which connect it to all nodes in the next layer
    for n in self.nodes:
      n._multiplyInputByWeights()      
      
  def _set_input(self, nodeValues):    
    for node, nodeValue in zip( self.nodes, nodeValues):
      node._setInput(nodeValue)
    
  def _get_neti(self,activationFunction):
    # print("_get_neti")    
    for n in self.nodes:
      n._setOutput(activationFunction(n._getInput()))
      self.pred.append(n.output)
      # print(n.output)      
  
  def sumMult(self): #obtain the input to all next layer nodes
    # print("total")
    i = 0
    self.total = 0 
    self.total = np.zeros((1,self.nodes[i].output[0].__len__()))
    while i < len(self.nodes):
      self.total += self.nodes[i].output
      i += 1
    # print(self.total)      
      

