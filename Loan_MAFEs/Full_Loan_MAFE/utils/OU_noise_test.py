#from noise import OUNoise
#import numpy as np
#from matplotlib import pyplot as plt
#
#
#dt = 0.01
#
#theta = 0.15 # friction: strength to pull towards the mean
#sigma = 0.3 # noise
#mu = 0.0 # global mean
#
#processes = 3
#samples = 1000
#
#
#zero = np.zeros((1000))
#
#for aa in range(5):
#    exploration = OUNoise((1,2), scale=0.01, sigma=sigma, mu=mu, theta=theta, dt=dt)
#    X = np.zeros(shape=(1000, 2))
#    for t in range(samples):
#        X[t,:] = exploration.noise()
#
#    #plt.figure(figsize=(16, 9))
#    plt.plot(X[:,0])
#    plt.plot(zero)
#    plt.title('Ornstein-Uhlenbeck Process (5 Trials)')
#plt.show()


from noise import OUNoise
import numpy as np
from matplotlib import pyplot as plt
import pdb


dt = 0.01

theta = 0.9 # friction: strength to pull towards the mean
sigma = 0.3 # noise
scale = 0.015
#mu_init = np.array([[0.0,0.0,0.0,0.0,0.0]])
#sig_init = np.array([[0.025,0.025,0.025,0.025,0.025]])
#mu = np.random.normal(mu_init,sig_init)
mu = np.array([[0.0,0.0,0.0,0.0,0.0]])
state = np.array([[0.05,-0.1,0.02,-0.04,0.0]])
mu = state
#processes = 5
samples = 800


zero = np.zeros((800))

exploration = OUNoise(mu.shape, scale=scale, sigma=sigma, mu=mu, theta=theta, dt=dt, state=state)
X = np.zeros(shape=(800, 5))
for t in range(samples):
    X[t,:] = exploration.noise()

#plt.figure(figsize=(16, 9))
plt.plot(X)
plt.plot(zero)
plt.title('Ornstein-Uhlenbeck Process (5 Trials)')
plt.legend([str(mu[0,0]), str(mu[0,1]), str(mu[0,2]), str(mu[0,3]), str(mu[0,4])])
plt.show()

