import numpy as np
import matplotlib.pyplot as plt

def generate_mu_sigma(mu1, sigma1 ,mu2, sigma2, clusters):
    mu_x = np.random.normal(mu1, sigma1, clusters)
    mu_y = np.random.normal(mu1, sigma1, clusters)
    sigma_x = np.random.normal(mu2, sigma2, clusters)
    sigma_y = np.random.normal(mu2, sigma2, clusters)
    mu = []
    sigma = []
    for i in range(clusters):
        mu.append([mu_x[i], mu_y[i]])
        sigma.append([[sigma_x[i], 0], [0, sigma_x[i]]])
    return (mu, sigma)
print generate_mu_sigma(1, 0.5, 1, 1.0, 4)

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

print generate_z([0.1, 0.2, 0.5, 0.2])

def generate_x(theta, mu1, mu2, sigma1, sigma2, clusters, num_points = 1000):
    mu, sigma = generate_mu_sigma(mu1, sigma1, mu2, sigma2, clusters)
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
    
print generate_x([0.1, 0.2, 0.5, 0.2], 1.0, 5.0, 10.0, 2.0, 4)
