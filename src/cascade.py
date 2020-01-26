import sys, os
import numpy as np
import pandas as pd
import pickle
from activation_functions import *
sys.path.append(
	os.path.join(
		os.path.dirname(__file__), 
		'..', 
		'utils'
	)
)
from util import *
import Padronizar
from ErroPercentualAbsoluto import *
import Arquivo

from sklearn.metrics import r2_score
from sklearn.preprocessing   import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn                 import preprocessing

class Cascade(object):

  def __init__(self, numMaxHiddenNodes):
    self.ensemble = dict.fromkeys(list(range(numMaxHiddenNodes)))
    self.weightsDic = dict.fromkeys(list(range(numMaxHiddenNodes)))
    self.residualError = dict.fromkeys(list(range(numMaxHiddenNodes)))

    self.X_train=[]
    self.X_test=[]
    self.y_train=[]
    self.y_test=[]
    print("construtor")

  def load_and_preprocess_data(self):

    data = Arquivo.ler()
    x, y = Padronizar.dividir(data, 60, 12)
    minmaxscaler = MinMaxScaler(feature_range=(0,1))
    dataNX, listMin,  listMax  = Padronizar.normalizarLinear(x, 0.1, 0.9)
    dataNY, listMin1, listMax2 = Padronizar.normalizarLinear(y, 0.1, 0.9)
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataNX, dataNY, train_size = 0.8, test_size  = 0.2)
    
    self.X_train = addBias(self.X_train.values)
    self.X_test  = addBias(self.X_test.values)
    # WE WILL USE ONLY THE FIRST COLUMN 														
    self.y_train = self.y_train[self.y_train.columns[0]].values
    self.y_test = self.y_test[self.y_test.columns[0]].values

  def insertHiddenUnit(self):
    wh = self.init_weights(self.X_train[0].__len__(),1)
    self.weightsDic[0] = wh
    # print(wh.shape)
    return self.forward(wh)

  def forward(self,wh):
    ent = np.dot( self.X_train,wh)
    neti = sigmoid(ent)
    return neti
  
  def init_weights(self,xcol,hiddennodes):    
    return (np.random.rand(xcol,hiddennodes))

  def calculateResidualError(self,w0,neti):
    calc = neti.T*w0
    print("Erro percentual médio absoluto")
    print(mean_absolute_percentage_error(self.y_train,calc[0]))
  
  def saveModel(self):
    print("salvar o modelo")