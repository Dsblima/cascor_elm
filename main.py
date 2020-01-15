import numpy as np
from elm import *

if __name__ == '__main__':
    elm = ELM()
    wh = elm.init_weights(21,5)
    wh = elm.insert_bias(wh)
    #print(wh)

    x = np.random.rand(123,20)
    x = elm.insert_bias(x)

    net = x.dot(wh)
    netInv = elm.pinv(net)
    print(netInv)