import numpy as np
import matplotlib.pyplot as plt

def generate_means(mu1, sigma1 , clusters):
    mu_out = []
    sigma_out = []
    for i in range(clusters-1):
        mu_1, mu1 = np.random.multivariate_normal(mu1, sigma1, 2)
        print mu_1
        sigma1 = np.array(sigma1)*0.5
        mu_out.append(mu_1)
    mu_out.append(mu1)
    return mu_out

print generate_means([2,2], [[10, 0],[0, 10]], 4)

def generate_variances(mu1, sigma1 , clusters):
    sigma_out = []
    for i in range(clusters-1):
        var_1, mu1 = np.random.normal(mu1, sigma1, 2) 
        sigma_out.append([[var_1, 0], [0, var_1]])
    sigma_out.append([[mu1, 0], [0, mu1]])
    return sigma_out
print generate_variances(5.0, 10.0, 4)

def generate_z(theta):
    z = []
    z_end=0
    for i in theta:
        z.append(z_end + i)
        z_end = z_end + i
    x  = np.random.random_sample()
    sample = 0
    for i in z:
        if x < i:
            break
        sample += 1
    return sample

print generate_z([0.15, 0.15, 0.2, 0.5])

def generate_x(theta, mu1, mu2, sigma1, sigma2, clusters, num_points = 1000):
    mu = generate_means(mu1, sigma1, clusters)
    print mu
    sigma = [[[5, 0],[0, 5]], [[2, 0],[0, 2]], [[1, 0],[0, 1]], [[0.5, 0],[0, 0.5]]]
    xs = []
    ys = []
    color = ['b', 'g', 'r', 'y']
    for i in range(num_points):
        z = generate_z(theta)
        x, y = np.random.multivariate_normal(mu[z], sigma[z], 1).T
        plt.plot(x, y, 'x', c=color[z])
        xs.append(x)
        ys.append(y)
    # plt.plot(xs, ys, 'x')
    plt.axis('equal')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    # plt.title('lamda = ' + str(lamda) + 'sigma = ' + str(sigma))
    plt.show()
    
print generate_x([0.4, 0.3, 0.2, 0.1], [5,5] , 1.0 , [[100.0, 0], [0, 100.0]], 4.0, 4)
