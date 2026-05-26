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

        # True replay buffer (reset)
        self.reset_replay_buffer()

        # Temporary buffer to hold candidates to add to replay buffer (reset)
        self.reset_candidates()
        
        self.filled_i = 0  # index of first empty location in buffer (last index when full)

    def reset_replay_buffer(self):

        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_true_buffs = []
        self.next_obs_fake_buffs = []
        self.empty_obs_buffs = []
        self.next_empty_obs_buffs = []
        self.dones_buffs = []

        for nn in range(self.num_agents):
            self.obs_buffs.append([])
            self.ac_buffs.append([])
            self.rew_buffs.append([])
            self.next_obs_true_buffs.append([])
            self.next_obs_fake_buffs.append([])
            self.empty_obs_buffs.append([])
            self.next_empty_obs_buffs.append([])
            self.dones_buffs.append([])
        
    def reset_candidates(self):

        self.cand_obs_buffs = []
        self.cand_ac_buffs = []
        self.cand_rew_buffs = []
        self.cand_next_obs_true_buffs = []
        self.cand_next_obs_fake_buffs = []
        self.cand_empty_obs_buffs = []
        self.cand_next_empty_obs_buffs = []
        self.cand_dones_buffs = []

        for nn in range(self.num_agents):
            self.cand_obs_buffs.append([])
            self.cand_ac_buffs.append([])
            self.cand_rew_buffs.append([])
            self.cand_next_obs_true_buffs.append([])
            self.cand_next_obs_fake_buffs.append([])
            self.cand_empty_obs_buffs.append([])
            self.cand_next_empty_obs_buffs.append([])
            self.cand_dones_buffs.append([])

    def __len__(self):
        return self.filled_i

    def sub_samp_push(self, samp_sz=80):

        ep_len = len(self.cand_obs_buffs[0])				 					# # # # #
				 										# # # # #
        samp_idx = np.random.choice(np.arange(ep_len, dtype=int),  size=np.min([samp_sz,ep_len]),     		# # # # # 
                                replace=False)					 				# # # # #
        									 				# # # # #
        if self.filled_i >= self.max_steps:				 					# # # # #
				 										# # # # #
            for agent_i in range(self.num_agents):				 				# # # # #
														# # # # #
                self.obs_buffs[agent_i] = self.obs_buffs[agent_i][samp_sz:]				 	# # # # #
                self.ac_buffs[agent_i] = self.ac_buffs[agent_i][samp_sz:]				 	# # # # #
                self.rew_buffs[agent_i] = self.rew_buffs[agent_i][samp_sz:]				 	# # # # #
                self.next_obs_true_buffs[agent_i] = self.next_obs_true_buffs[agent_i][samp_sz:]				 	# # # # #
                self.next_obs_fake_buffs[agent_i] = self.next_obs_fake_buffs[agent_i][samp_sz:]				 	# # # # #
                self.empty_obs_buffs[agent_i] = self.empty_obs_buffs[agent_i][samp_sz:]				 		# # # # #
                self.next_empty_obs_buffs[agent_i] = self.next_empty_obs_buffs[agent_i][samp_sz:]				# # # # #
                self.dones_buffs[agent_i] = self.dones_buffs[agent_i][samp_sz:]				 			# # # # #
				 												# # # # #
            for agent_i in range(self.num_agents):				 				 		# # # # #
                self.obs_buffs[agent_i].extend([self.cand_obs_buffs[agent_i][ind] for ind in samp_idx])				 		# # # # #
                self.ac_buffs[agent_i].extend([self.cand_ac_buffs[agent_i][ind] for ind in samp_idx])				 		# # # # #
                self.rew_buffs[agent_i].extend([self.cand_rew_buffs[agent_i][ind] for ind in samp_idx])  # Each agent receives same reward.	# # # # #
                self.next_obs_true_buffs[agent_i].extend([self.cand_next_obs_true_buffs[agent_i][ind] for ind in samp_idx])			# # # # #
                self.next_obs_fake_buffs[agent_i].extend([self.cand_next_obs_fake_buffs[agent_i][ind] for ind in samp_idx])			 # # # # #
                self.empty_obs_buffs[agent_i].extend([self.cand_empty_obs_buffs[agent_i][ind] for ind in samp_idx])				 # # # # #
                self.next_empty_obs_buffs[agent_i].extend([self.cand_next_empty_obs_buffs[agent_i][ind] for ind in samp_idx])			# # # # #
                self.dones_buffs[agent_i].extend([self.cand_dones_buffs[agent_i][ind] for ind in samp_idx])				 	# # # # #
				 												# # # # #
        else:															 # # # # #
																# # # # #
            for agent_i in range(self.num_agents):										# # # # #
                self.obs_buffs[agent_i].extend([self.cand_obs_buffs[agent_i][ind] for ind in samp_idx])					 	# # # # #
                self.ac_buffs[agent_i].extend([self.cand_ac_buffs[agent_i][ind] for ind in samp_idx])						# # # # #
                self.rew_buffs[agent_i].extend([self.cand_rew_buffs[agent_i][ind] for ind in samp_idx])  # Each agent receives same reward.      # # # # #
                self.next_obs_true_buffs[agent_i].extend([self.cand_next_obs_true_buffs[agent_i][ind] for ind in samp_idx])			# # # # #
                self.next_obs_fake_buffs[agent_i].extend([self.cand_next_obs_fake_buffs[agent_i][ind] for ind in samp_idx])			# # # # #
                self.empty_obs_buffs[agent_i].extend([self.cand_empty_obs_buffs[agent_i][ind] for ind in samp_idx])				# # # # #
                self.next_empty_obs_buffs[agent_i].extend([self.cand_next_empty_obs_buffs[agent_i][ind] for ind in samp_idx])			# # # # #
                self.dones_buffs[agent_i].extend([self.cand_dones_buffs[agent_i][ind] for ind in samp_idx])				 # # # # #
                													# # # # #
        self.filled_i = len(self.obs_buffs[agent_i])									# # # # #

    def push(self):

        ep_len = len(self.cand_obs_buffs[0])
        
        if self.filled_i >= self.max_steps:

            for agent_i in range(self.num_agents):

                self.obs_buffs[agent_i] = self.obs_buffs[agent_i][ep_len:]
                self.ac_buffs[agent_i] = self.ac_buffs[agent_i][ep_len:]
                self.rew_buffs[agent_i] = self.rew_buffs[agent_i][ep_len:]
                self.next_obs_true_buffs[agent_i] = self.next_obs_true_buffs[agent_i][ep_len:]
                self.next_obs_fake_buffs[agent_i] = self.next_obs_fake_buffs[agent_i][ep_len:]
                self.empty_obs_buffs[agent_i] = self.empty_obs_buffs[agent_i][ep_len:]
                self.next_empty_obs_buffs[agent_i] = self.next_empty_obs_buffs[agent_i][ep_len:]
                self.dones_buffs[agent_i] = self.dones_buffs[agent_i][ep_len:]

            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i].extend(self.cand_obs_buffs[agent_i])
                self.ac_buffs[agent_i].extend(self.cand_ac_buffs[agent_i])
                self.rew_buffs[agent_i].extend(self.cand_rew_buffs[agent_i])  # Each agent receives same reward.
                self.next_obs_true_buffs[agent_i].extend(self.cand_next_obs_true_buffs[agent_i])
                self.next_obs_fake_buffs[agent_i].extend(self.cand_next_obs_fake_buffs[agent_i])
                self.empty_obs_buffs[agent_i].extend(self.cand_empty_obs_buffs[agent_i])
                self.next_empty_obs_buffs[agent_i].extend(self.cand_next_empty_obs_buffs[agent_i])
                self.dones_buffs[agent_i].extend(self.cand_dones_buffs[agent_i])

        else:

            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i].extend(self.cand_obs_buffs[agent_i])
                self.ac_buffs[agent_i].extend(self.cand_ac_buffs[agent_i])
                self.rew_buffs[agent_i].extend(self.cand_rew_buffs[agent_i])  # Each agent receives same reward.
                self.next_obs_true_buffs[agent_i].extend(self.cand_next_obs_true_buffs[agent_i])
                self.next_obs_fake_buffs[agent_i].extend(self.cand_next_obs_fake_buffs[agent_i])
                self.empty_obs_buffs[agent_i].extend(self.cand_empty_obs_buffs[agent_i])
                self.next_empty_obs_buffs[agent_i].extend(self.cand_next_empty_obs_buffs[agent_i])
                self.dones_buffs[agent_i].extend(self.cand_dones_buffs[agent_i])
                
        self.filled_i = len(self.obs_buffs[agent_i])

    def push_candidates(self, observations, actions, rewards, next_observations_true, next_observations_fake, empty_obs, next_empty_obs, dones):

        for agent_i in range(self.num_agents):
            self.cand_obs_buffs[agent_i].append(observations[agent_i])
            self.cand_ac_buffs[agent_i].append(actions[agent_i])
            self.cand_rew_buffs[agent_i].append(rewards)  # Each agent receives same reward.
            self.cand_next_obs_true_buffs[agent_i].append(next_observations_true[agent_i])
            self.cand_next_obs_fake_buffs[agent_i].append(next_observations_fake[agent_i])
            self.cand_empty_obs_buffs[agent_i].append(empty_obs[agent_i])
            self.cand_next_empty_obs_buffs[agent_i].append(next_empty_obs[agent_i])
            self.cand_dones_buffs[agent_i].append(dones)

    def sample(self, N, to_gpu=False, norm_rews=True):

        inds = np.random.choice(np.arange(self.filled_i), size=N,      
                                replace=False)                         

        return ([[self.obs_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.ac_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.rew_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.next_obs_true_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.next_obs_fake_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.empty_obs_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.next_empty_obs_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                [[self.dones_buffs[i][ind] for ind in inds] for i in range(self.num_agents)])
                







