import sys, os
import numpy as np
import cascade
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

def elmTeste():
    x = np.random.randint(60,size=(123,20))
    x = addBias(x)
    y = np.random.randint(60,size=(123,1))
    
    elm = ELM(5,x,y)
    wh = elm.init_weights(x[0].__len__(),elm.hiddennodes)

    net = x.dot(wh)
    net = sigmoid(net)

    netInv = elm.pinv(x.dot(wh))

    w0 = elm.getW0(netInv)
    print(w0)

def cascadeTeste():
    num_hidden_nodes = 200
    hiddennodes = list(range(num_hidden_nodes))
    neti = np.zeros((num_hidden_nodes,1))
    cascade: Cascade = Cascade(num_hidden_nodes)
    cascade.load_and_preprocess_data()
    for i in hiddennodes:
        print(i)
        neti= cascade.insertHiddenUnit(i)
        elm = ELM(1,cascade.X_train,cascade.y_train)
        netInv = elm.pinv(neti)
        w0 = elm.getW0(netInv)
        cascade.calculateResidualError(w0,neti)

if __name__ == '__main__':
    cascadeTeste()
    
