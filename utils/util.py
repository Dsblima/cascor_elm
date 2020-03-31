import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from sklearn.metrics import mean_squared_error
import Arquivo
from DataHandler import *
import Padronizar
from sklearn.model_selection import train_test_split

def add_1(arr):
    return np.append(arr,1)
  
def addBias(matrix):#this function adds the value 1, at the last position in each array from matrix
    
  bias = np.ones((matrix.__len__(),1))
  # return np.array(list(map(add_1, matrix)))
  return np.concatenate((matrix,bias),axis=1)

def roundNumber(number):
  return round(number,4)

def printErrors(mape,mse):
    print("mean_absolute_percentage_error")
    print(mape)
    print("mean_squared_error")
    print(mse)
    print()
    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
  
def load_and_preprocess_data(baseName,dimension):
    
    data = Arquivo.ler('../data/'+baseName)
    dh:DataHandler = DataHandler(data, dimension, 60, 20,20)
    X_train, y_train, val_set, val_target, X_test, y_test, arima_train, arima_val, arima_test= dh.redimensiondata(data, 12, 60, 20,20)

    y = [[]]
    x = np.concatenate( (X_train,X_test) )   

    y = np.matrix( np.concatenate( (np.array(y_train),np.array(y_test)) ))   
    data = np.concatenate((x, y.T), axis=1)
    x, y = Padronizar.dividir(data, 12, 1)


    # data = Arquivo.ler('../data/projeto base completa.csv')
    # x, y = Padronizar.dividir(data, 4, 1)
    # print("y")
    # print(y[y.columns[0]])
    # minmaxscaler = MinMaxScaler(feature_range=(0,1))
    dataNX, listMin,  listMax  = Padronizar.normalizarLinear(x, 0.1, 0.9)
    dataNY, listMinY, listMaxY = Padronizar.normalizarLinear(y, 0.1, 0.9)
    # scalerX,scalerY, dataNormalizadoX, dataNormalizadoY = Padronizar.normalizar(x,y)
    X_train, X_test, y_train, y_test = train_test_split(dataNX, dataNY, train_size = 0.8, test_size  = 0.2)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size = 0.5, test_size  = 0.5)
    
    X_train = addBias(X_train.values)    
    X_val  = addBias(X_val.values)
    X_test  = addBias(X_test.values)
    
    # WE WILL USE ONLY THE FIRST COLUMN 														
    y_train = y_train[y_train.columns[0]].values
    y_test = y_test[y_test.columns[0]].values
    y_val = y_val[y_val.columns[0]].values
    
    return   X_train, y_train, X_val, y_val, X_test, y_test

def calculateResidualError(trueValues,pred):    
    # print(np.matrix(trueValues).T)
    # trueValues = Padronizar.desnormalizarLinear(pd.DataFrame(np.matrix(trueValues).T), self.listMaxY, self.listMinY, 0.1, 0.9)
    # print(trueValues)
    # pred = Padronizar.desnormalizarLinear(pd.DataFrame(np.matrix(pred)), self.listMaxY, self.listMinY, 0.1, 0.9)
    # print(pred)
    # print("Residual Error")
    mape = mean_absolute_percentage_error(trueValues,pred)
    
    mse = mean_squared_error(trueValues,pred)    
    
    return mape, mse

def plot():
  print("plot")
  # df=pd.DataFrame({'x': range(1,num_hidden_nodes+1), 'y1': cascade.residualError, 'y2': cascade.residualErrorTest})
        
  # plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='grey', markersize=12, color='grey', linewidth=4,label='Validação')
  # plt.plot( 'x', 'y2', data=df, marker='', markerfacecolor='black', markersize=12, color='black', linewidth=4,label='Teste')
  
  # # Add a legend
  # plt.legend()
  # plt.show()
  # Plot the data
  # plt.plot(y, cascade.residualError, label='MSE')

  # # Add a legend
  # plt.legend()
  
  # plt.savefig('200 hiddeunits.png')  
    