import sys, os
import numpy as np
from cascade import *
from elm import *
from activation_functions import * 

sys.path.append(
	os.path.join(
		os.path.dirname(__file__), 
		'..', 
		'utils'
	)
)
from util import *
from cascade import *
from DataHandler import *
import Padronizar
from ErroPercentualAbsoluto import *
import Arquivo
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def elmExecute():

    # LOAD AND PROCESS DATA
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
    X_train, X_test, y_train, y_test = train_test_split(dataNX, dataNY, train_size = 0.8, test_size  = 0.2)    
    X_train = addBias(X_train.values)     
    X_test  = addBias(X_test.values)
    y_train = y_train[y_train.columns[0]].values
    y_test = y_test[y_test.columns[0]].values

    # Instance and run ELM
    elm = ELM(20,X_train,y_train)
    wh = elm.init_weights(X_train[0].__len__(),elm.hiddennodes)
    net = X_train.dot(wh)
    net = sigmoid(net)
    netInv = elm.pinv(X_train.dot(wh))
    w0 = elm.getW0(netInv)
    
    # Calculate the metrics
    pred = net.dot(w0.T)
    print("mean_absolute_percentage_error")
    print(mean_absolute_percentage_error(y_train,pred))
    print("mean_squared_error")
    print(mean_squared_error(y_train,pred))
    print("r2_score")
    print(r2_score(y_train,pred))

def cascadeExecute():
    num_hidden_nodes = 50
    hiddennodes = list(range(num_hidden_nodes))
    # neti = np.zeros((num_hidden_nodes,1))
    cascade: Cascade = Cascade(num_hidden_nodes)
    cascade.load_and_preprocess_data()
    cascade.fit(cascade.X_train,cascade.y_train)
           
    cascade.predict(cascade.X_test,cascade.y_test)
    
    
    df=pd.DataFrame({'x': range(1,num_hidden_nodes+1), 'y1': cascade.residualError, 'y2': cascade.residualErrorTest})
        
    plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='grey', markersize=12, color='grey', linewidth=4,label='Validação')
    plt.plot( 'x', 'y2', data=df, marker='', markerfacecolor='black', markersize=12, color='black', linewidth=4,label='Teste')
    
    # Add a legend
    plt.legend()
    plt.show()
    # Plot the data
    # plt.plot(y, cascade.residualError, label='MSE')

    # # Add a legend
    # plt.legend()
    
    # plt.savefig('200 hiddeunits.png')

if __name__ == '__main__':
    cascadeExecute()
    
