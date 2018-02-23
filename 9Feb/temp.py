from GMM_Classifier import *
import numpy as np


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
    for d in [10]: #d
        for n in [100, 1000, 10000]: #n
            for k in [2, 5, 10 ]: #k                
                X = np.loadtxt("data/data_points_d{}_n{}_k{}".format(d, n, k))
                Y  = np.loadtxt("data/label_points_d{}_n{}_k{}".format(d, n, k))
                print d,n,k
                # X = np.loadtxt("data/d_{}n_{}k_{}.txt".format(d, n, k), delimiter=',')
                try:
                    # mu = X[np.random.choice(n, k, False), :]
                    # sigma = [np.eye(d) for i in range(k)]                    
                    # EM_gaussian(X, mu, sigma, theta[k])
                    c = GMM_Clustering(X, k)
                    c.sigma = c.sigma/10.0
                    q,w,e = c.EM_gaussian()
                    # print q
                    # print w
                    # print e
                    output = c.predict()
                    print c.Rand_index(Y, output)
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
                
