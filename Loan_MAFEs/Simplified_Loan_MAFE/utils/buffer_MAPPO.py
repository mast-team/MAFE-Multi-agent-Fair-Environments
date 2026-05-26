import numpy as np
from torch import Tensor
from torch.autograd import Variable
from utils.data_structures import SegmentTree, MinSegmentTree, SumSegmentTree
import random
import pdb

class ReplayBuffer(object):

    # Currently no parallel or gpu support. To be added
    
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self):
        """
        Inputs:
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.num_agents = 3

        # True replay buffer (reset)
        self.reset_replay_buffer()

        # Temporary buffer to hold candidates to add to replay buffer (reset)
        self.reset_candidates()
        
        self.filled_i = 0  # index of first empty location in buffer (last index when full)

    def reset_replay_buffer(self):

        self.obs_buffs = []
        self.ac_buffs = []
        self.Gs_buffs = []
        self.log_probs_buffs = []
        self.rew_buffs = []
        self.empty_obs_buffs = []
        self.dones_buffs = []

        for nn in range(self.num_agents):
            self.obs_buffs.append([])
            self.ac_buffs.append([])
            self.Gs_buffs.append([])
            self.log_probs_buffs.append([])
            self.rew_buffs.append([])
            self.empty_obs_buffs.append([])
            self.dones_buffs.append([])
        
    def reset_candidates(self):

        self.cand_obs_buffs = []
        self.cand_ac_buffs = []
        self.cand_Gs_buffs= []
        self.cand_log_probs_buffs = []
        self.cand_rew_buffs = []
        self.cand_empty_obs_buffs = []
        self.cand_dones_buffs = []

        for nn in range(self.num_agents):
            self.cand_obs_buffs.append([])
            self.cand_ac_buffs.append([])
            self.cand_Gs_buffs.append([])
            self.cand_log_probs_buffs.append([])
            self.cand_rew_buffs.append([])
            self.cand_empty_obs_buffs.append([])
            self.cand_dones_buffs.append([])

    def __len__(self):
        return self.filled_i

    def push(self):

        ep_len = len(self.cand_obs_buffs[0])
        
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i].extend(self.cand_obs_buffs[agent_i])
            self.ac_buffs[agent_i].extend(self.cand_ac_buffs[agent_i])
            self.Gs_buffs[agent_i].extend(self.cand_Gs_buffs[agent_i])
            self.log_probs_buffs[agent_i].extend(self.cand_log_probs_buffs[agent_i])
            self.rew_buffs[agent_i].extend(self.cand_rew_buffs[agent_i])  # Each agent receives same reward.
            self.empty_obs_buffs[agent_i].extend(self.cand_empty_obs_buffs[agent_i])
            self.dones_buffs[agent_i].extend(self.cand_dones_buffs[agent_i])
                
        self.filled_i = len(self.obs_buffs[agent_i])

    def push_candidates(self, observations, actions, log_probs, rewards, empty_obs, dones):

        for agent_i in range(self.num_agents):
            self.cand_obs_buffs[agent_i].append(observations[agent_i])
            self.cand_ac_buffs[agent_i].append(actions[agent_i])
            self.cand_log_probs_buffs[agent_i].append(log_probs[agent_i])
            self.cand_rew_buffs[agent_i].append(rewards)  # Each agent receives same reward.
            self.cand_empty_obs_buffs[agent_i].append(empty_obs[agent_i])
            self.cand_dones_buffs[agent_i].append(dones)

    # ADJUST TO NEVER REPEAT SELECTION
    def sample(self, N, to_gpu=False, norm_rews=True):

        inds = np.random.choice(np.arange(self.filled_i), size=N,      
                                replace=False)                         

        return ([[self.obs_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.ac_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.Gs_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.log_probs_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.rew_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.empty_obs_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.dones_buffs[i][ind] for ind in inds] for i in range(self.num_agents)])
                
