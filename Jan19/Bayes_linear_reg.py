import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from math import sqrt
def joindata(datafile):
    i = 0
    for dataf in datafile:
        data1 = np.loadtxt(dataf, delimiter=',')
        if i == 0:
            data = data1
            i = 1
            continue
        data = np.append(data, data1, axis=0)
    return data

def BaysianLinear(data, lamda = 0.5, sigma= 0.5):
    y = data.transpose()[53:54].transpose()
    x = data.transpose()[0:53]
    term1 = (sigma/2.0) * np.matmul(x,x.transpose())
    term2 = (lamda/2.0) * np.identity(term1.shape[0])
    w =  np.matmul(inv(term1 + term2 ), ((sigma/2)*np.matmul(x,y)))
    return w

def Test(data, w):
    y = data.transpose()[53:54].transpose()
    x = data.transpose()[0:53]
    Y_pred = np.matmul(x.transpose(), w)
    err =0
    for i in range(y.shape[0]):
        err = err + (y[i]- Y_pred[i])**2
    return sqrt(float(err)/y.shape[0])

def LinearRegression(lamda = 0.5, sigma = 0.5):
    data = joindata(["Dataset/Dataset/Training/Features_Variant_1.csv",
                     "Dataset/Dataset/Training/Features_Variant_2.csv",
                     "Dataset/Dataset/Training/Features_Variant_3.csv",
                     "Dataset/Dataset/Training/Features_Variant_4.csv",
                     "Dataset/Dataset/Training/Features_Variant_5.csv"])
    data1, data2, data3, data4, data5 = np.array_split(data, 5)
    data1 =  data1
    data2 =  np.append(data1, data2, axis=0)
    data3 =  np.append(data2, data3, axis=0)
    data4 =  np.append(data3, data4, axis=0)
    err = []
    w = BaysianLinear(data1, lamda, sigma)
    err.append(Test(data5, w))
    w = BaysianLinear(data2, lamda, sigma)
    err.append(Test(data5, w))
    w = BaysianLinear(data3, lamda, sigma)
    err.append(Test(data5, w))
    w = BaysianLinear(data4, lamda, sigma)
    err.append(Test(data5, w))
    print err
    ploterr(err, [data1.shape[0], data2.shape[0], data3.shape[0], data4.shape[0]], lamda, sigma)
    
def ploterr(err, datasize, lamda, sigma):
    plt.plot(datasize, err, marker='o', c='r')
    plt.xlabel('size of data')
    plt.ylabel('square root error')
    plt.title('lamda = ' + str(lamda) + 'sigma = ' + str(sigma))
    plt.show()
    
LinearRegression(lamda = 0.1, sigma = 0.1)
