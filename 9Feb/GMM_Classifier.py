import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import inv, det
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product
class GMM_Clustering:
    def __init__(self, X, K=2, theta=None):
        self.k = K
        self.X = np.array(X)
        self.n, self.d = X.shape
        if not theta:
            theta = [1./self.k] * self.k
        self.theta = theta
        self.mu = X[np.random.choice(self.n, self.k, False), :]
        self.sigma = np.array([np.eye(self.d) for i in range(self.k)])
        self.respon = np.zeros((self.n, self.k))
        self.log_likelihoods = []
    def EM_gaussian(self, eps=0.001, max_iltr = 1000):
        log_lik = 0.0
        while len(self.log_likelihoods) < max_iltr:
            old_lik = log_lik
            try:
                log_lik = self.E_step()
            except:
                break
            if np.any(np.isnan(self.respon)):
                print "NAN"
                print self.log_likelihoods, log_lik
                break
            ### Likelihood computation
        
            self.log_likelihoods.append(log_lik)
        
            self.M_step()
            # check for convergence
            if len(self.log_likelihoods) < 2 : continue
            if np.abs(log_lik - self.log_likelihoods[-2]) < eps: break
        return (self.mu, self.sigma, self.theta)

    def E_step(self):
        # P = lambda mu, s: det(s) ** -.5 ** (2 * np.pi) ** (-self.d/2.)* np.exp(-.5 * np.einsum('ij, ij -> i', self.X - mu, np.dot(inv(s) , (self.X - mu).T).T ) )
        # d = len(theta)   
        for k in range(self.k):
            # print sigma[k]
            var = multivariate_normal(self.mu[k], self.sigma[k], allow_singular=True)
            self.respon[:, k] = var .pdf(self.X) # / var.pdf(self.mu[k]) #self.theta[k] * P(self.mu[k], self.sigma[k])
        log_likelihood = np.sum(np.log(np.sum(self.respon, axis = 1)))
        self.respon = (self.respon.T / np.sum(self.respon, axis = 1)).T
        return log_likelihood

    def M_step(self):
        ## The number of datapoints belonging to each gaussian            
        N_ks = np.sum(self.respon, axis = 0)
        mu_n = self.mu
        sigma_n = self.sigma
        theta_n = self.theta
        for k in range(self.k):
            ## means
            mu_n[k] = 1. / N_ks[k] * np.sum(self.respon[:, k] * self.X.T, axis = 1).T
            x_mu = np.matrix(self.X - mu_n[k])
            if not np.any(np.isnan(mu_n)):
                self.mu = mu_n
            ## covariances
            sigma_n[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  self.respon[:, k]), x_mu))
            if not np.any(np.isnan(sigma_n)):
                self.sigma = sigma_n
            ## and finally the probabilities
            theta_n[k] = (1. / self.n) * N_ks[k]
            if not np.any(np.isnan(theta_n)):
                self.theta = theta_n
        return self.mu, self.sigma, self.theta
    def predict(self, X=None):
        if not X:
            X=self.X
        # P = lambda mu, s: det(s) ** -.5 ** (2 * np.pi) ** (-self.X.shape[1]/2.)* np.exp(-.5 * np.einsum('ij, ij -> i', self.X - mu, np.dot(inv(s) , (self.X - mu).T).T ) ) 
        for k in range(self.k):
            var = multivariate_normal(self.mu[k], self.sigma[k])
            self.respon[:, k] = var.pdf(X) 
            # self.respon[:, k] = self.theta[k] * P(self.mu[k], self.sigma[k])
        self.respon = (self.respon.T / np.sum(self.respon, axis = 1)).T
        pred = np.argmax(self.respon, axis=1)
        pred = map(lambda x: x+1, pred)
        return np.argmax(self.respon, axis=1)
    def Rand_index(self, Y_actual, Y_pred):
        a = 0.0
        b = 0.0
        c = 0.0
        d = 0.0
        temp = range(len(self.X))
        for i,j in product(temp,temp):
            if Y_actual[i]==Y_actual[j] and Y_pred[i]==Y_pred[j]:
                a += 1
            if Y_actual[i]!=Y_actual[j] and Y_pred[i]!=Y_pred[j]:
                b += 1
            if Y_actual[i]==Y_actual[j] and Y_pred[i]!=Y_pred[j]:
                c += 1
            if Y_actual[i]!=Y_actual[j] and Y_pred[i]==Y_pred[j]:
                d += 1
        return (float(a+b)/float(a+b+c+d))
                

if __name__=="__main__":
    var = {2:{},
           5:{},
           10:{}}
    theta1 = [0.5, 0.5]
    theta2 = [0.2, 0.2, 0.2, 0.2, 0.2]
    theta3 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    theta = {2 : theta1,
             5 : theta2,
             10 : theta3}
    for d in [2, 5, 10]: #d
        for n in [100, 1000, 10000]: #n
            for k in [2, 5, 10]: #k                
                X = np.loadtxt("data/data_points_d{}_n{}_k{}".format(d, n, k))
                Y  = np.loadtxt("data/label_points_d{}_n{}_k{}".format(d, n, k))
                print d,n,k
                # X = np.loadtxt("data/d_{}n_{}k_{}.txt".format(d, n, k), delimiter=',')
                try:
                    # mu = X[np.random.choice(n, k, False), :]
                    # sigma = [np.eye(d) for i in range(k)]                    
                    # EM_gaussian(X, mu, sigma, theta[k])
                    c = GMM_Clustering(X, k)
                    q,w,e = c.EM_gaussian()
                    # print q
                    # print w
                    # print e
                    output = c.predict()
                    try:
                        print c.Rand_index(Y, output)
                    except:
                        print "error in rand"
                except:
                    print "SOME ERROR OCCURED"
                # if d==2:
                #     xs, ys = X.T
                #     colors = cm.rainbow(np.linspace(0, 1, 10))
                #     label =  colors[output, :]
                #     plt.scatter(xs, ys, s=4,color=label)
                #     # plt.plot(xs, ys, 'x')
                #     plt.axis('equal')
                #     plt.xlabel('x_1')
                #     plt.ylabel('x_2')
                #     # plt.title('lamda = ' + str(lamda) + 'sigma = ' + str(sigma))
                #     plt.show()                    
                
                    
