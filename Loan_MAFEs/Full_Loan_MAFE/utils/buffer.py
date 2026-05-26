import numpy as np
from torch import Tensor
from torch.autograd import Variable
import pdb

class ReplayBuffer(object):

    # Currently no parallel or gpu support. To be added
    
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = 3
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_true_buffs = []
        self.next_obs_fake_buffs = []
        self.empty_obs_buffs = []
        self.next_empty_obs_buffs = []
        self.done_buffs = []
        for nn in range(self.num_agents):
            self.obs_buffs.append([])
            self.ac_buffs.append([])
            self.rew_buffs.append([])
            self.next_obs_true_buffs.append([])
            self.next_obs_fake_buffs.append([])
            self.empty_obs_buffs.append([])
            self.next_empty_obs_buffs.append([])

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations_true, next_observations_fake, empty_obs, next_empty_obs):

        pdb.set_trace()
        if self.curr_i == self.max_steps:
            pdb.set_trace()
            rollover = self.max_steps - 1 - self.curr_i # num of indices to roll over (should be -1)
            pdb.set_trace()
            for agent_i in range(self.num_agents):

                pdb.set_trace()
                self.obs_buffs[agent_i] = self.obs_buffs[agent_i][1:]
                self.ac_buffs[agent_i] = self.ac_buffs[agent_i][1:]
                self.rew_buffs[agent_i] = self.rew_buffs[agent_i][1:]
                self.next_obs_true_buffs[agent_i] = self.next_obs_true_buffs[agent_i][1:]
                self.next_obs_fake_buffs[agent_i] = self.next_obs_fake_buffs[agent_i][1:]
                self.empty_obs_buffs[agent_i] = self.empty_obs_buffs[agent_i][1:]
                self.next_empty_obs_buffs[agent_i] = self.next_empty_obs_buffs[agent_i][1:]

            pdb.set_trace()
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i].append(observations[agent_i])
                self.ac_buffs[agent_i].append(actions[agent_i])
                self.rew_buffs[agent_i].append(rewards)  # Each agent receives same reward.
                self.next_obs_true_buffs[agent_i].append(next_observations_true[agent_i])
                self.next_obs_fake_buffs[agent_i].append(next_observations_fake[agent_i])
                self.empty_obs_buffs[agent_i].append(empty_obs[agent_i])
                self.next_empty_obs_buffs[agent_i].append(next_empty_obs[agent_i])
            self.curr_i = self.max_steps-1
            self.filled_i = self.max_steps
        else:
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i].append(observations[agent_i])
                self.ac_buffs[agent_i].append(actions[agent_i])
                self.rew_buffs[agent_i].append(rewards)  # Each agent receives same reward.
                self.next_obs_true_buffs[agent_i].append(next_observations_true[agent_i])
                self.next_obs_fake_buffs[agent_i].append(next_observations_fake[agent_i])
                self.empty_obs_buffs[agent_i].append(empty_obs[agent_i])
                self.next_empty_obs_buffs[agent_i].append(next_empty_obs[agent_i])
            self.filled_i += 1

        self.curr_i += 1
        

    def sample(self, N, to_gpu=False, norm_rews=True):

        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False)
 
        return ([[self.obs_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.ac_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.rew_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.next_obs_true_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.next_obs_fake_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.empty_obs_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.next_empty_obs_buffs[i][ind] for ind in inds] for i in range(self.num_agents)])
                
