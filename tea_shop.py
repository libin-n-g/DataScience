'''
Develop a sequentially learning system for a tea-shop to provide 
personalized service to its valued customers. The system will 
predict whether a customer will prefer tea or coffee in a 
particular day. 

coffee => 0
tea => 1
'''
import numpy as np
import numpy.random as d  
import matplotlib.pyplot as plt  

def updatebeta(a,b,Y):
    if Y == 'coffee':
        return (a, 1+b)
    if Y == 'tea':
        return (a+1, b)
def Bayesian(Y='coffee'):
    a=0.5
    b=50.0
    indicater = {'coffee':0, 'tea':1}
    correct=[indicater[Y] for i in range(100)]
    predicted=[]
    for i in range(100):
        X = d.beta(a,b)
        if X < 0.5:
            print 'COFFEE'
            predicted.append(indicater['coffee'])
        else :
            print 'TEA'
            predicted.append(indicater['tea'])
        (a,b) = updatebeta(a,b,Y)
        print (a,b)
    plt.plot(range(100), correct, 'bs', label='actual', markersize=7)
    plt.plot(range(100), predicted,'ro',  label='predicted')
    plt.title('Bayesian sequential predictor (coffee => 0, tea => 1)')
    plt.legend(loc=4)
    plt.show()
Bayesian(Y='tea')
