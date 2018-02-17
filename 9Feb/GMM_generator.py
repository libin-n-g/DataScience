import numpy as np

def generate_z(theta):
    z = []
    z_end=0
    for i in theta:
        z.append(z_end + i)
        z_end = z_end + i
        if z_end > 1:
            raise ValueError
    x  = np.random.random_sample()
    sample = 0
    for i in z:
        if x < i:
            break
        sample += 1
    return sample
def generate_mu_sigma (d, mu_, sigma_):
    sigma = [[0 for i in range (d)] for j in range(d)]
    sigma_temp = np.random.normal(2.0, 0.5, 1)
    for i in range(d):
        sigma[i][i] = sigma_temp[0]
    mu = np.random.multivariate_normal(mu_, np.array(sigma_), 1)
    return (list(mu), sigma)
    
def generate_x(theta, mu, sigma, n = 1000):
    out = []
    label = []
    for i in range(n):
        z = generate_z(theta)
        sample_point = np.random.multivariate_normal(list(mu[z]), sigma[z], 1)
        label.append([z])
        out.append(sample_point)
    return (np.array(out), np.array(label))
def write3d(filename ,data, flag=1):
    data = np.array(data)
    with file(filename, 'w+') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        if flag:
            outfile.write('# Array shape: {0}\n'.format(data.shape))
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:
            
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            if flag:
                outfile.write('# New slice\n')
if __name__=='__main__':
    mu  = { 2: {},
            5: {},
            10: {}}
    sigma =  { 2: {},
               5: {},
               10: {}}
    mu_ = { 2: [0.0, 0.0],
            5: [0.0, 0.0, 0.0, 0.0, 0.0],
            10:[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    sigma_ = { 2: [[20.0, 0.0],[0.0, 20.0]],
               5: [[20.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 20.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 20.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 20.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 20.0]],
               10: [[20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0]]}
    for d in [2, 5, 10]: #d
        print "dim =" + str(d)
        for k in [2,5,10]:
            print "K= " + str(k)
            ms = []
            ss = []
            for i in range(k):
                m, s = generate_mu_sigma (d, mu_[d], sigma_[d])
                ms.extend(m)
                ss.append(s)
            mu[d][k] = ms
            sigma[d][k] = ss
    theta1 = [0.3, 0.7]
    theta2 = [0.1, 0.15, 0.2, 0.25, 0.3]
    theta3 = [0.05, 0.15, 0.05, 0.1, 0.2, 0.1, 0.1, 0.05, 0.15, 0.05]
    theta = {2 : theta1,
             5 : theta2,
             10 : theta3}
    out = {}
    label = {}
    i=0
    for d in [2, 5, 10]: #d
        for n in [100, 1000, 10000]: #n
            for k in [2, 5, 10]: #k
                out[i], label[i] =  generate_x(theta[k], mu[d][k], sigma[d][k], n)
                write3d("parm/parm_mean_d{}_n{}_k{}".format(d, n, k) , mu[d][k])
                write3d("parm/parm_var_d{}_n{}_k{}".format(d, n, k) , sigma[d][k])
                write3d("data/data_points_d{}_n{}_k{}".format(d, n, k) , out[i], 0)
                write3d("data/label_points_d{}_n{}_k{}".format(d, n, k) , label[i], 0)
                i = i+1

