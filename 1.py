import random
import numpy as np
import matplotlib.pyplot as plt
def gaussion_distribution(mu=0, sigma=1.0, num= 100):
    x = []
    for i in range(100):
        x.append(random.gauss(mu,sigma))
    return x

def d2D_guassian(mu=0, sigma=1.0, num= 100):
    return (gaussion_distribution(mu, sigma, num),
            gaussion_distribution(mu, sigma, num))


x1,y2 =  d2D_guassian()
x, y = np.random.multivariate_normal([0.0, 0.0], [[10.0,0.0],[0.0,1.0]], 100).T
plt.scatter(x1,y2, c='r')
plt.scatter(x,y, c='g')
plt.show()

