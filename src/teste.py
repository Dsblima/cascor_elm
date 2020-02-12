from Layer import *
from cascadeMLP import *
from activation_functions import *
import numpy as np
import pandas as pd
import sys,os

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
  
if __name__ == "__main__":
  
  cascadeMLP: CascadeMLP = CascadeMLP(learningRate=0.05, ciclos = 200)
  cascadeMLP.load_and_preprocess_data()
  cascadeMLP.init_layers()
  cascadeMLP.training()
  
  y = list(range(cascadeMLP.ciclos))
  # print(y)
  # Plot the data
  plt.plot(y, cascadeMLP.mse, label='MSE')

  # Add a legend
  plt.legend()

  # Show the plot
  plt.show()     
  
  # ok 1 - initiate the network/read data
  # ok 2 - initiate the layers
  # ok 3 - initiate the weights
  # ok train
  # ok 4 - load the first line from the training set
  # ok 5 - set the input nodes with these values
  # ok 6 multiply each node value for all weights which connect them to the hidden layer
  # ok 7 - obtain the sum of the multiplications, that is what value will be the input to the each hidden layer node
  # ok 8 - apply the actovation function to each hidden node value
  # ok 9 - multiply each node value for all weights which connect them to the output layer
  # ok 10 - obtain the sum of the multiplications, that is what value will be the input to the each output layer node
  # ok 11 - apply the actovation function to each output node value to obtain the output
  # ok 12 - calculate the residual ErroPercentualAbsoluto
  # ok 13 - adjust the weights which connect hidden and output layer
  # 14 - execute the steps from 4 to 13 until no significant error reductions happen   
  # 15 - if the current network performance is satisfactory stop the training, otherwise add another hidden unit.
   
  # print("input layer")
  # inputLayer._get_output() # multiplies each node input for all weights which connect it to the next layer
  # inputLayer.sumMult()# obtain the input to the next layer, summing the multiplication of all input for the weights
  
  # print("hidden layer")
  # hiddenLayer._set_input(inputLayer.total[0])
  # hiddenLayer._get_neti(sigmoid)
  # print("neti*weights")
  
  # hiddenLayer._get_output()
  # hiddenLayer.sumMult()
  
  # print("Output Layer")
  # outputLayer._set_input(hiddenLayer.total[0])
  # outputLayer._get_neti(linear)
  
  