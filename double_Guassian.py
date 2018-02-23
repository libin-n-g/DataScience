'''
Sample from two Gaussian distributions with means m1, and m2 respectively 
and s.d. s1, and s2 respectively. 
Assume dimensionality d=2, and dist(m1-m2) is D. 
Vary,  s1, s2, and D and generate 5 different plots.
'''
import matplotlib.pyplot as plt
import numpy as np
def PlotGuassian1D(mu=0, sigma=0.1):
    s = np.random.normal(mu, sigma, 1000)
    return s
    
def generatedoubleGuassian(sigma1=0.5, sigma2=0.5, D=0.0):
    mu1 = (D/2.0)
    mu2 = (-D/2.0)
    print mu1, mu2
    x1 = PlotGuassian1D(mu1, sigma1)
    y1 = PlotGuassian1D(mu1, sigma1)
    x2 = PlotGuassian1D(mu2, sigma2)
    y2 = PlotGuassian1D(mu2, sigma2)
    plt.scatter(x1,y1, label=' mean='+str(mu1)+' sigma='+str(sigma1))
    plt.scatter(x2,y2, label=' mean='+str(mu2)+' sigma='+str(sigma2))
    plt.title('Guassian Distribution with D='+str(D))
    plt.legend(loc=2)
    plt.show()

generatedoubleGuassian()
