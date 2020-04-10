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
    mseArray = []
    # num_hidden_nodes_array = list(range(1,101,1))
    # for num_hidden_nodes in num_hidden_nodes_array:
    elm = ELM(80,)
    elm.fit(X_train,y_train)
        
    pred =elm.pred(X_test)

    mape, mse, rmse  = calculateResidualError(y_test,pred)      
    # mseArray.append(mse)
    del elm
    # print(mseArray)    
    # plot(baseName,"ELM",100,mseArray,[],[],"MSE",'','',False,True)    
    
    return mape,mse    

def cascadeExecute(baseName,dimension):
    num_hidden_nodes = 40        
    cascade: Cascade = Cascade(num_hidden_nodes)
    cascade.X_train, cascade.y_train, cascade.X_val, cascade.y_val, cascade.X_test, cascade.y_test= load_and_preprocess_data(baseName,dimension)
    cascade.fit(cascade.X_train,cascade.y_train)
    
    # plot(baseName,num_hidden_nodes,cascade.mapeArray,[],[],"MAPE",'','',True)
    
    predTeste = cascade.predict(cascade.X_test)
    
    mape, mse,rmse  = calculateResidualError(cascade.y_test, predTeste)
    cascade.mapeArrayTest.append(mape)
    cascade.mseArrayTest.append(mse)
    
    del cascade
           
    return mape,mse


if __name__ == '__main__':    
    
    bases = ["airlines2", "Monthly Sunspot Dataset", "Minimum Daily Temperatures Dataset", "Daily Female Births Dataset",'Colorado River','Eletric','Gas','Lake Erie','Pollution','redwine']
    dimensions = [12,11,12,12,12,12,12,12,12,12]
    
    # bases = ["Monthly Sunspot Dataset"]
    # dimensions = [11]    
    for base, dimension in zip(bases, dimensions):
        mapeListCascade = []
        mseListCascade = []
        mapeListELM = []
        mseListELM = []
        for i in range(1,31):        
            mape,mse = cascadeExecute(base,dimension)
            mapeListCascade.append(mape)
            mseListCascade.append(mse)
                    
            mape,mse = elmExecute(base, dimension)
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
    	