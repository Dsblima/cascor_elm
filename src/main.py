import sys, os
import csv
import numpy as np
from cascade import *
from elm import *
from datetime import date
sys.path.append(
	os.path.join(
		os.path.dirname(__file__), 
		'..', 
		'utils'
	)
)
from Chart import *
from util import *
from activation_functions import *
from JsonManager import *

def executeELM(today, bases,upperLimit, lowerLimit, dimensions, maxHiddenNodes, minHiddenNodes, iterations, model):
        
    lambdaValues = [1, 10, 100, 1000, 10000, 100000]
    for base, dimension in zip(bases, dimensions):
        # for lambdaValue in lambdaValues:
        # folderToSave = today+' lambda = '+ str(lambdaValue)
        folderToSave = today+' ELM Construtivo'
        dictToSave = {}
        
        print(base)
        mseTestByNumHiddenNodesList = []
        mapeTestByNumHiddenNodesList = []
        mseValByNumHiddenNodesList = []
        mapeValByNumHiddenNodesList = []
        predValList = []
        targetValList = []
        predTestList = []
        targetTestList = []
        listNodes = list(range(minHiddenNodes, maxHiddenNodes+1))
        
        for num_hidden_nodes in listNodes:
            print (base+' - '+str(num_hidden_nodes))
            mseTestListELM = []
            mapeTestListELM = []
            mseValListELM = []
            mapeValListELM = []
                    
            for i in list(range(1, iterations+1)):        
                elm = ELM(num_hidden_nodes)
                X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(base,dimension, lowerLimit, upperLimit)
                elm.fit(X_train,y_train)        
                
                predTest = elm.pred(X_test)
                mapeTest, mseTest,rmseTest  = calculateResidualError(y_test, predTest)
                predVal = elm.pred(X_val)
                mapeVal, mseVal,rmseVal  = calculateResidualError(y_val, predVal)
                
                # cascade.mapeArrayTest.append(mape)
                # mseArray.append(mse)
                
                mapeTestListELM.append(mapeTest)
                mseTestListELM.append(mseTest)
                mapeValListELM.append(mapeVal)
                mseValListELM.append(mseVal)
                
                predValList.append(predVal)
                targetValList.append(y_val)
                predTestList.append(predTest)
                targetTestList.append(y_test)
                
            mseTestByNumHiddenNodesList.append(np.mean(mseTestListELM))
            mapeTestByNumHiddenNodesList.append(np.mean(mapeTestListELM))
            mseValByNumHiddenNodesList.append(np.mean(mseValListELM))
            mapeValByNumHiddenNodesList.append(np.mean(mapeValListELM))    
        
        dictToSave['model'] = model
        dictToSave['activationFunction'] = 'sigmoid'
        dictToSave['inputsize']  = dimension
        dictToSave['executions'] = []
        
        for mapeValValue, mseValValue, mapeTestValue, mseTestValue, valPredValues, valTargetValues, testPredValues, testTargetValues, numHiddenNodes in zip( mapeValByNumHiddenNodesList, mseValByNumHiddenNodesList,mapeTestByNumHiddenNodesList, mseTestByNumHiddenNodesList,  predValList, targetValList, predTestList, targetTestList, listNodes ):
            
            dictToSave['executions'].append(
                {
                    "numHiddenNodes":numHiddenNodes,
                    "predVal":valPredValues.tolist(),
                    "trueVal":valTargetValues.tolist(),
                    "predTest":testPredValues.tolist(),
                    "trueTest":testTargetValues.tolist(),
                    "errors":[
                        {
                            "mapeVal":mapeValValue,
                            "mseVal":mseValValue,
                            "rmseVal":"0",
                            "mapeTest":mapeTestValue,
                            "mseTest":mseTestValue,
                            "rmseTest":"0"
                        }
                    ]
                }
            )
            
        writeJsonFile(dictToSave, base, folderToSave)    

def executeCascade(today, bases,upperLimit, lowerLimit, dimensions, maxHiddenNodes, minHiddenNodes, iterations, model):
    # num_hidden_nodes = 40
    # mseArray = []
    # num_hidden_nodes_array = list(range(1,101,1))
    lambdaValues = [1, 10, 100, 1000, 10000, 100000]
    lambdaValue = 1
    trainingModel =  'cconecandidate'
    regularization = False
    folderToSave = today+trainingModel
    
    for base, dimension in zip(bases, dimensions):
        # for lambdaValue in lambdaValues:
            # folderToSave = today+' lambda = '+ str(lambdaValue)
        dictToSave = {}
        
        print(base)
        mseTestByNumHiddenNodesList = []
        mapeTestByNumHiddenNodesList = []
        mseValByNumHiddenNodesList = []
        mapeValByNumHiddenNodesList = []
        predValList = []
        targetValList = []
        predTestList = []
        targetTestList = []
        listNodes = list(range(minHiddenNodes, maxHiddenNodes+1))
        
        for num_hidden_nodes in listNodes:
            print (base+' - '+str(num_hidden_nodes))
            mseTestListCascade = []
            mapeTestListCascade = []
            mseValListCascade = []
            mapeValListCascade = []
                    
            for i in list(range(1, iterations+1)):        
                cascade: Cascade = Cascade(num_hidden_nodes, lambdaValue)
                cascade.X_train, cascade.y_train, cascade.X_val, cascade.y_val, cascade.X_test, cascade.y_test= load_and_preprocess_data(base,dimension, lowerLimit, upperLimit)
                cascade.fit(cascade.X_train,cascade.y_train, trainingModel, regularization)        
                
                predTest = cascade.predict(cascade.X_test)
                mapeTest, mseTest,rmseTest  = calculateResidualError(cascade.y_test, predTest)
                predVal = cascade.predict(cascade.X_val)
                mapeVal, mseVal,rmseVal  = calculateResidualError(cascade.y_val, predVal)
  
                mapeTestListCascade.append(mapeTest)
                mseTestListCascade.append(mseTest)
                mapeValListCascade.append(mapeVal)
                mseValListCascade.append(mseVal)
                
                predValList.append(predVal)
                targetValList.append(cascade.y_val)
                predTestList.append(predTest)
                targetTestList.append(cascade.y_test)
                
            mseTestByNumHiddenNodesList.append(np.mean(mseTestListCascade))
            mapeTestByNumHiddenNodesList.append(np.mean(mapeTestListCascade))
            mseValByNumHiddenNodesList.append(np.mean(mseValListCascade))
            mapeValByNumHiddenNodesList.append(np.mean(mapeValListCascade))    
        
        dictToSave['model'] = model
        dictToSave['activationFunction'] = 'sigmoid'
        dictToSave['inputsize']  = dimension
        dictToSave['executions'] = []
        
        for mapeValValue, mseValValue, mapeTestValue, mseTestValue, valPredValues, valTargetValues, testPredValues, testTargetValues, numHiddenNodes in zip( mapeValByNumHiddenNodesList, mseValByNumHiddenNodesList,mapeTestByNumHiddenNodesList, mseTestByNumHiddenNodesList,  predValList, targetValList, predTestList, targetTestList, listNodes ):
            
            dictToSave['executions'].append(
                {
                    "numHiddenNodes":numHiddenNodes,
                    "predVal":valPredValues.tolist(),
                    "trueVal":valTargetValues.tolist(),
                    "predTest":testPredValues.tolist(),
                    "trueTest":testTargetValues.tolist(),
                    "errors":[
                        {
                            "mapeVal":mapeValValue,
                            "mseVal":mseValValue,
                            "rmseVal":"0",
                            "mapeTest":mapeTestValue,
                            "mseTest":mseTestValue,
                            "rmseTest":"0"
                        }
                    ]
                }
            )
            
        writeJsonFile(dictToSave, base, folderToSave)                    
            
    # plot(baseName,"CASCADE",100,mseArray,[],[],"MSE",'','',False,True)        
    # return mape,mse 
    # chart.plotTable(mseValArray,filename+'MSEVal.csv')

if __name__ == '__main__':
    bases = ["airlines2", "Daily Female Births Dataset",'Colorado River','Eletric','Gas','Lake Erie','Pollution','redwine', "Monthly Sunspot Dataset", "Minimum Daily Temperatures Dataset"]
    dimensions = [12,12,12,12,12,12,12,12,11,12]
    bases = ['airlines2']
    dimensions = [12]
    upperLimit = 1
    lowerLimit = 0
    maxHiddenNodes = 70
    minHiddenNodes = 1
    iterations = 30    
    model = "Cascade - Elm - IELM"
    today = str(date.today())            
    executeCascade(today, bases, upperLimit, lowerLimit, dimensions, maxHiddenNodes, minHiddenNodes, iterations, model)    
    # executeELM(today, bases, upperLimit, lowerLimit, dimensions, maxHiddenNodes, minHiddenNodes, iterations, model)    
    titles = []
    saveChart = False
    showChart = True   
    dir1 = '../data/simulations/'+today+' qr fatorization'+'/'
    dir2 = '../data/simulations/2020-06-20 lambda = 10/'
    dir3 = '../data/simulations/2020-06-20 lambda = 100/'
    dir4 = '../data/simulations/2020-06-20 lambda = 1000/'
    dir5 = '../data/simulations/2020-06-20 lambda = 10000/'
    dir6 = '../data/simulations/2020-06-20 lambda = 100000/'
    # dirs = [dir1, dir2, dir3, dir4, dir5, dir6]
    dirs = [dir1]
    titles.append("Constructive ELM") 
    titles.append(" with lambda = 100000")
    folderToSave = "10000 x 100000/"
    # titles = [title1, title2]
    # visualizeResults(bases, dirs, titles, model, saveChart, showChart, folderToSave, maxHiddenNodes)	
    
    for folder in dirs:
        print(folder)
        getNumHiddenNodesSmallestError(bases, folder, model)
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')