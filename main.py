import numpy as np
from elm import *
from util import *

if __name__ == '__main__':
    x = np.random.randint(60,size=(123,20))
    x = addBias(x)
    y = np.random.randint(60,size=(123,1))
    
    elm = ELM(5,x,y)
    wh = elm.init_weights(x[0].__len__(),elm.hiddennodes)

    #net = x.dot(wh)
    netInv = elm.pinv(x.dot(wh))

    w0 = elm.getW0(netInv)
    print(w0)