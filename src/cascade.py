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
from DataHandler import *
import Padronizar
from ErroPercentualAbsoluto import *
import Arquivo

from sklearn.metrics import r2_score
from sklearn.preprocessing   import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn                 import preprocessing

class Cascade(object):

  def __init__(self, numMaxHiddenNodes = 800000):
    self.numMaxHiddenNodes = numMaxHiddenNodes
    self.ensemble = {}
    self.weightsArray = {}
    self.residualError = []

    self.X_train=[]
    self.X_test=[]
    self.y_train=[]
    self.y_test=[]

  def load_and_preprocess_data(self):
    # TESTE COM A BASE airlines 2
    # data = Arquivo.ler('../data/airlines2.txt')
    # dh:DataHandler = DataHandler(data, 12, 60, 20,20)
    # self.X_train, self.y_train, val_set, val_target, self.X_test, self.y_test, arima_train, arima_val, arima_test= dh.redimensiondata(data, 12, 60, 20,20)

    # y = [[]]
    # x = np.concatenate( ((self.X_train),(self.X_test)) ) 
    # y = np.matrix( np.concatenate( (np.array(self.y_train),np.array(self.y_test)) ))
    # data = np.concatenate((x, y.T), axis=1)
    # x, y = Padronizar.dividir(data, 12, 1)


    data = Arquivo.ler('../data/projeto base completa.csv')
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

  def init_weights(self,xcol):    
    return (np.random.rand(xcol,1))

  def insertHiddenUnit(self,i):
    numCol = self.X_train[0].__len__()+i
    # print(numCol)
    wh = self.init_weights(numCol)
    # print("weightsArray shape")
    # print(wh.shape)

    # if i==0:
    #   self.weightsArray = wh
    # else:
      # print("weightsArray shape")
      # print(self.weightsArray.shape)
      # self.weightsArray = np.matrix(np.concatenate( (self.weightsArray,wh), axis=1))
    self.weightsArray[i] = wh
    # print(self.weightsArray)
    # sys.exit(-1)
    
    return self.forward(self.weightsArray)

  def forward(self,wh):
    netis = [[]]
    x_train = self.X_train
    for node, weights in self.weightsArray.items():
      if node == 0:
        
        ent = np.dot( x_train,weights)
        neti = sigmoid(ent)      
        netis = neti
      else:
        # print(neti.shape)
        x_train = np.concatenate((x_train,neti),axis=1)
        ent = np.dot( x_train,weights)
        neti = sigmoid(ent)      
        netis = np.concatenate((netis,neti), axis=1)
        # print(x_train.shape)      
    netis = addBias(np.matrix(netis))
    print(neti.shape)
    print(netis.shape)
    # sys.exit(-1)
    # print("netis.shape")
    # print(netis[1:].shape)    
    # print(neti)
    return netis

  def calculateResidualError(self,w0,neti):
    preds = []
    calc = neti.dot(w0.T)

    # print(calc)
    print("Erro percentual médio absoluto")
    print(mean_absolute_percentage_error(self.y_train,calc))
    print("r2_score")
    print(r2_score(self.y_train,calc))
  
  def saveModel(self):
    print("salvar o modelo")