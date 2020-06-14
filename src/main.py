import sys, os
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
from activation_functions import * 
from util import *
from Chart import *
from util import *
from activation_functions import *
from JsonManager import *

def elmExecute(baseName,dimension):
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data(baseName,dimension)

    # Instance and run ELM
    mseArray = []
    num_hidden_nodes_array = list(range(1,101,1))
    for num_hidden_nodes in num_hidden_nodes_array:
        elm = ELM(num_hidden_nodes,)
        elm.fit(X_train,y_train)
            
        pred =elm.pred(X_test)

        mape, mse, rmse  = calculateResidualError(y_test,pred)      
        mseArray.append(mse)
        del elm
    # print(mseArray)    
    plot(baseName,"ELM",100,mseArray,[],[],"MSE",'','',False,True)    
    
    return mape,mse    

def executeCascade(today, bases,upperLimit, lowerLimit, dimensions, maxHiddenNodes, minHiddenNodes, iterations, model):
    # num_hidden_nodes = 40
    # mseArray = []
    # num_hidden_nodes_array = list(range(1,101,1))
    lambdaValues = [1, 10, 100, 1000, 10000, 100000]
    for base, dimension in zip(bases, dimensions):
        for lambdaValue in lambdaValues:
            folderToSave = today+' lambda = '+ str(lambdaValue)
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
                    cascade.X_train, cascade.y_train, cascade.X_val, cascade.y_val, cascade.X_test, cascade.y_test= load_and_preprocess_data(base,dimension)
                    cascade.fit(cascade.X_train,cascade.y_train)        
                    
                    predTest = cascade.predict(cascade.X_test)
                    mapeTest, mseTest,rmseTest  = calculateResidualError(cascade.y_test, predTest)
                    predVal = cascade.predict(cascade.X_val)
                    mapeVal, mseVal,rmseVal  = calculateResidualError(cascade.y_val, predVal)
                    
                    # cascade.mapeArrayTest.append(mape)
                    # mseArray.append(mse)
                    
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
def getErrors(dictToRead):
    
    mapeValArray = []
    mseValArray = []
    mapeTestArray = []
    mseTestArray = []
    for execution in dictToRead['executions']:
        
        for error in execution['errors']:
            mapeValArray.append(error['mapeVal'])
            mseValArray.append(error['mseVal'])
            mapeTestArray.append(error['mapeTest'])
            mseTestArray.append(error['mseTest'])
    
    return mapeValArray, mseValArray, mapeTestArray, mseTestArray

def getPredAndTrueValues(dictToRead, node):
    predVal = []
    trueVal = []
    predTest = []
    trueTest = []
    for execution in dictToRead['executions']:
        if execution['numHiddenNodes'] == node:
            predVal = execution['predVal']
            trueVal = execution['trueVal']
            predTest = execution['predTest']
            trueTest = execution['trueTest']

    return predVal, trueVal, predTest, trueTest

def visualizeResults(bases, dirs, titles, model, saveChart, showChart, folderToSave):
    
    for base in bases:
        mseVal = []
        mseTest = []
        chart:Chart = Chart()
        
        for folder, title in zip(dirs, titles):
        
            fileName = folder+base
        
            loadedDict = readJsonFile(fileName+'.json')    
            
            mapeValArray, mseValArray, mapeTestArray, mseTestArray = getErrors(loadedDict)
            mseVal.append(mseValArray)
            mseTest.append(mseTestArray)
            predVal, trueVal, predTest, trueTest = getPredAndTrueValues(loadedDict,2)
            
        chart.plotValidationAndTest(base, model, maxHiddenNodes, mseVal,
                                mseTest, "Validation", "Test", title1, title2, showChart, saveChart, folderToSave) 
    # chart.plotTable(mseValArray,filename+'MSEVal.csv')

if __name__ == '__main__':
    bases = ["airlines2", "Daily Female Births Dataset",'Colorado River','Eletric','Gas','Lake Erie','Pollution','redwine', "Monthly Sunspot Dataset", "Minimum Daily Temperatures Dataset"]
    dimensions = [12,12,12,12,12,12,12,12,11,12]
    # bases = ['Minimum Daily Temperatures Dataset']
    # dimensions = [12]
    upperLimit = 1
    lowerLimit = -1
    maxHiddenNodes = 70
    minHiddenNodes = 1
    iterations = 1    
    today = str(date.today())            
    # executeCascade(today, bases, upperLimit, lowerLimit, dimensions, maxHiddenNodes, minHiddenNodes, iterations, model)    
    
    model = "Cascade - ARIMA"
    saveChart = True
    showChart = False   
    dir1 = '../data/simulations/2020-06-12 lambda = 10000/'
    dir2 = '../data/simulations/2020-06-12N regularization/'
    dirs = [dir1, dir2]
    title1 = " with lambda = 10000" 
    title2 = " without regularization"
    folderToSave = "reg x no reg/"
    titles = [title1, title2]
    visualizeResults(bases, dirs, titles, model, saveChart, showChart, folderToSave)	