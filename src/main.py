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

def elmExecute(baseName,dimension):
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(baseName,dimension)

    # Instance and run ELM
    elm = ELM(500,)
    elm.fit(X_train,y_train)
        
    pred =elm.pred(X_test)
   
    mape, mse  = calculateResidualError(y_test,pred)      
    
    # printErrors(mape,mse)
    return mape,mse    

def cascadeExecute(baseName,dimension):
    num_hidden_nodes = 100        
    cascade: Cascade = Cascade(num_hidden_nodes)
    cascade.X_train, cascade.y_train, cascade.X_val, cascade.y_val, cascade.X_test, cascade.y_test= load_and_preprocess_data(baseName,dimension)
    cascade.fit(cascade.X_train,cascade.y_train)
           
    predTeste = cascade.predict(cascade.X_test)
    mape, mse  = calculateResidualError(cascade.y_test, predTeste)
    cascade.mapeArrayTest.append(mape)
    cascade.residualErrorTest.append(mse)
    
    # printErrors(mape,mse)        
    return mape,mse

if __name__ == '__main__':    
    
    bases = ['airlines2.txt','Minimum Daily Temperatures Dataset.txt','Monthly Sunspot Dataset.txt','Daily Female Births Dataset.txt']
    dimensions = [12,12,11,12]
    for base,dimension in zip(bases,dimensions):
        mapeListCascade = []
        mseListCascade = []
        mapeListELM = []
        mseListELM = []
        for i in range(1,31):        
            mape,mse = cascadeExecute(base,dimension)
            mapeListCascade.append(mape)
            mseListCascade.append(mse)
                    
        for i in range(1,31):
            mape,mse = elmExecute(base,dimension)
            mapeListELM.append(mape)
            mseListELM.append(mse)
        
        print(base)   
        print("cascade")
        print(np.mean(mapeListCascade))   
        print(np.mean(mseListCascade))
        print("elm")   
        print(np.mean(mapeListELM))   
        print(np.mean(mseListELM))
        print()   
    
