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

from Layer import *
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing   import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn                 import preprocessing

class CascadeMLP(object):

  def __init__(self, learningRate=0.05, ciclos=10):
    
    self.ensemble = {}
    self.weightsArray = {}
    self.mse = []
    self.mapeArray = []
    
    self.learningRate = learningRate
    self.ciclos = ciclos

    self.X_train=[]
    self.X_test=[]
    self.y_train=[]
    self.y_test=[]
    self.menor = 100000000
    # self.dataNX=[]
    # self.listMin=[]
    # self.listMax=[]    
    # self.dataNY=[]
    # self.listMinY=[]
    # self.listMaxY=[]
    
  def adjustWeights(self,erro):
    for n in self.hiddenLayer.nodes:
      # print("old weights")
      # print(n.weightsArray)
      n.weightsArray = n.weightsArray + self.learningRate*erro*n.output
      # print("new weights")
      # print(n.weightsArray)
    # print("Wnew = Wold + lr*E*x")
  
    
  def load_and_preprocess_data(self):
    
    data = Arquivo.ler('../data/airlines2.txt')
    dh:DataHandler = DataHandler(data, 12, 60, 20,20)
    X_train, y_train, val_set, val_target, X_test, y_test, arima_train, arima_val, arima_test= dh.redimensiondata(data, 12, 60, 20,20)
    y = [[]]
    x = np.concatenate( ((X_train),(X_test)) ) 
    y = np.matrix( np.concatenate( (np.array(y_train),np.array(y_test)) ))
    data = np.concatenate((x, y.T), axis=1)
    x, y = Padronizar.dividir(data, 12, 1)
    dataNX, listMin,  listMax  = Padronizar.normalizarLinear(x, 0.1, 0.9)
    dataNY, listMinY, listMaxY = Padronizar.normalizarLinear(y, 0.1, 0.9)
    # scalerX,scalerY, dataNormalizadoX, dataNormalizadoY = Padronizar.normalizar(x,y)
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(dataNX, dataNY, train_size = 0.8, test_size  = 0.2)    
    self.X_train = addBias(self.X_train.values)     
    self.X_test  = addBias(self.X_test.values)
    self.y_train = self.y_train[self.y_train.columns[0]].values
    self.y_test = self.y_test[self.y_test.columns[0]].values    

  def init_weights(self):
    
    self.inputLayer._init_weights(len(self.hiddenLayer.nodes))# Initiate randomly the weights
    self.hiddenLayer._init_weights(len(self.outputLayer.nodes))# Initiate randomly the weights    
  def init_layers(self):
    self.inputLayer  = Layer(len(self.X_train[0]))
    self.hiddenLayer = Layer(1)
    self.outputLayer = Layer(1)
    
    # print(len(self.outputLayer.nodes))
    
    self.init_weights()

  def insertHiddenUnit(self,i):
    # numCol = self.X_train[0].__len__()+i
    
    # wh = self.init_weights(numCol)
    # ws = self.init_weights(numCol)

    # self.weightsArray[i] = wh
    print("to do")      

  def forward(self):
    # print("forward")
    # print("input layer")
    self.inputLayer._get_output() # multiplies each node input for all weights which connect it to the next layer
    self.inputLayer.sumMult()# obtain the input to the next layer, summing the multiplication of all input for the weights      
    # print("hidden layer")
    self.hiddenLayer._set_input(self.inputLayer.total[0])
    self.hiddenLayer._get_neti(sigmoid)
    # print("neti*weights")
    self.hiddenLayer._get_output()
    self.hiddenLayer.sumMult()
    # print("Output Layer")
    self.outputLayer._set_input(self.hiddenLayer.total[0])
    self.outputLayer._get_neti(linear)  
  
  def training(self):    
    ciclo = 0
    while ciclo < self.ciclos:
      i = 0
      trueValues = []
      predValues = []
      while i < len(self.X_train):
        self.outputLayer.pred = []
        data = self.X_train[i]
        self.inputLayer._set_input(data)
        self.forward()
        # print("Error")
        trueValues.append(self.y_train[i])
        predValues.append(self.outputLayer.pred)
        error = self.y_train[i]-self.outputLayer.pred        
        self.calculateResidualError(trueValues, predValues)
        self.adjustWeights(error)      
        # break
        i += 1
      ciclo += 1
      

  def calculateResidualError(self,trueValues,pred):    
    # print(np.matrix(trueValues).T)
    # trueValues = Padronizar.desnormalizarLinear(pd.DataFrame(np.matrix(trueValues).T), self.listMaxY, self.listMinY, 0.1, 0.9)
    # print(trueValues)
    # pred = Padronizar.desnormalizarLinear(pd.DataFrame(np.matrix(pred)), self.listMaxY, self.listMinY, 0.1, 0.9)
    # print(pred)
    # print("Residual Error")
    # mape = mean_absolute_percentage_error(trueValues,pred)
    # self.mapeArray.append(mape)
    # print("mean_absolute_percentage_error")
    # print(mape)
    # print("mean_squared_error")
    mse = mean_squared_error(trueValues,pred)
    self.mse.append(mse)
    print(mse)
    
    # print("r2_score")
    # print(r2_score(trueValues,pred))

  def saveModel(self):
    print("salvar o modelo")