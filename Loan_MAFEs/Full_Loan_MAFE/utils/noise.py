import numpy as np
import pdb


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2, dt=0.1, state=0.0):

        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = state
        self.dt = dt

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x)*self.dt + self.sigma * np.random.randn(*x.shape)
        self.state = x + dx*self.scale
        return self.state
