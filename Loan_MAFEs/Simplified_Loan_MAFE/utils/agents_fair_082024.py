from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from torch.distributions import Normal
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb
import numpy as np
from scipy.stats import norm
from torch.optim.lr_scheduler import LinearLR

########################################## AGENT OBJECTS ##########################################

class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64,
                 lr=0.01, onehot_dim=0):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
        """
        self.policy = DiscretePolicy(num_in_pol, num_out_pol,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim)
        self.target_policy = DiscretePolicy(num_in_pol,
                                            num_out_pol,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        return self.policy(obs, sample=explore)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])

#------------------------------Adaptation of MAAC------------------------------

class AdmissionAgent_MAAC(object):

    def __init__(self, num_in_pol, policyTypes, eq_flag, AwareOrNot,
                 input_dim_critic1, input_dim_critic2, input_dim_critic3,
                 hidden_dim, which_agents, lr=0.01):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            policy_type (int): form of policy actions
            num_in_critic (int): number of dimensions for critic input
        """

        self.policy = Admission_Policy_Networks_MAAC(num_in_pol, policyTypes, eq_flag, AwareOrNot)


        self.target_policy = Admission_Policy_Networks_MAAC(num_in_pol, policyTypes, eq_flag, AwareOrNot)

        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

        self.policy_loss = np.array([])
        self.critic_loss = np.array([])
        

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale):
        self.exploration.scale = scale

    def step(self, obs, step_num, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)

        # continuous action
        if explore:
            M,N = action.shape
            mu = np.zeros((M,N))

            # Gradually descrease exploration
            if step_num<15000:
                sig = np.ones((M,N))*0.05
            elif (step_num>=15000) and (step_num<25000):
                sig = np.ones((M,N))*0.025
            else:
                sig = np.ones((M,N))*0.01
                
            noise = np.random.normal(mu,sig)
            action += Variable(Tensor(noise),
                                requires_grad=False)
        action = action.clamp(0, 1)

        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class DisburseAgent_MAAC(object):

    def __init__(self, num_in_pol, policyTypes, eq_flag, AwareOrNot,
                 input_dim_critic1, input_dim_critic2, input_dim_critic3,
                 hidden_dim, which_agents, lr=0.01):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            policy_type (int): form of policy actions
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = Disburse_Policy_Networks_MAAC(num_in_pol, policyTypes, eq_flag, AwareOrNot)
                
        self.target_policy = Disburse_Policy_Networks_MAAC(num_in_pol, policyTypes, eq_flag, AwareOrNot)

        self.policy_loss = np.array([])
        self.critic_loss = np.array([])
        
        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale):
        self.exploration.scale = scale

    def step(self, obs, step_num, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)

        # continuous action
        if explore:
            M,N = action.shape
            mu = np.zeros((M,N))
            
            # Gradually descrease exploration
            if step_num<15000:
                sig = np.ones((M,N))*0.05
            elif (step_num>=15000) and (step_num<25000):
                sig = np.ones((M,N))*0.025
            else:
                sig = np.ones((M,N))*0.01
                
            noise = np.random.normal(mu,sig)
            action += Variable(Tensor(noise),
                                requires_grad=False)
        action = action.clamp(0, 1)
        
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class DebtAgent_MAAC(object):

    def __init__(self, num_in_pol, policyTypes, eq_flag, AwareOrNot,
                 input_dim_critic1, input_dim_critic2, input_dim_critic3,
                 hidden_dim, which_agents, lr=0.01):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            policy_type (int): form of policy actions
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = Debt_Policy_Networks_MAAC(num_in_pol, policyTypes, eq_flag, AwareOrNot)
                
        self.target_policy = Debt_Policy_Networks_MAAC(num_in_pol, policyTypes, eq_flag, AwareOrNot)

        self.policy_loss = np.array([])
        self.critic_loss = np.array([])
        
        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale):
        self.exploration.scale = scale

    def step(self, obs, step_num, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)

        # continuous action
        if explore:
            M,N = action.shape
            mu = np.zeros((M,N))
            
            # Gradually descrease exploration
            if step_num<15000:
                sig = np.ones((M,N))*0.05
            elif (step_num>=15000) and (step_num<25000):
                sig = np.ones((M,N))*0.025
            else:
                sig = np.ones((M,N))*0.01
                
            noise = np.random.normal(mu,sig)
            action += Variable(Tensor(noise),
                                requires_grad=False)
        action = action.clamp(0, 1)
        
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

#------------------------------------------------------------------------------------


class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

#------------------------------MAPPO------------------------------

class AdmissionAgent_MAPPO(object):
    def __init__(self, num_in_pol, policyTypes, eq_flag, AwareOrNot,
                 input_dim_critic1, input_dim_critic2, input_dim_critic3,
                 hidden_dim, which_agents, lr_critic=0.01, lr_policy=0.001):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            policy_type (int): form of policy actions
            num_in_critic (int): number of dimensions for critic input
        """

        #----------------------------------------Policy Network---------------------------------------
        self.policyTypes = policyTypes
        self.eq_flag = eq_flag
        self.AwareOrNot = AwareOrNot
        self.policy = Admission_Policy_Networks_MAPPO(num_in_pol, policyTypes, eq_flag, AwareOrNot)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_policy)
        self.policy_loss = np.array([])

        #--------------------------------------Critic Network--------------------------------------
        self.critic = Critic_Networks(input_dim_critic1, input_dim_critic2, input_dim_critic3, hidden_dim, which_agents)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        self.critic_loss = {}
        self.critic_loss['Main'] = np.array([])

    def step(self, obs):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        action = self.policy(obs)
 
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class DisburseAgent_MAPPO(object):

    def __init__(self, num_in_pol, policyTypes, eq_flag, AwareOrNot,
                 input_dim_critic1, input_dim_critic2, input_dim_critic3,
                 hidden_dim, which_agents, lr_critic=0.01, lr_policy=0.001):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            policy_type (int): form of policy actions
            num_in_critic (int): number of dimensions for critic input
        """
        
        #----------------------------------------Policy Network---------------------------------------
        self.policyTypes = policyTypes
        self.eq_flag = eq_flag
        self.AwareOrNot = AwareOrNot
        self.policy = Disburse_Policy_Networks_MAPPO(num_in_pol, policyTypes, eq_flag, AwareOrNot)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_policy)
        self.policy_loss = np.array([])

        #--------------------------------------Critic Network--------------------------------------
        self.critic = Critic_Networks(input_dim_critic1, input_dim_critic2, input_dim_critic3, hidden_dim, which_agents)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        self.critic_loss = {}
        self.critic_loss['Main'] = np.array([])

    def scale_noise(self, scale):
        self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        
        action = self.policy(obs)
        
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class DebtAgent_MAPPO(object):

    def __init__(self, num_in_pol, policyTypes, eq_flag, AwareOrNot,
                 input_dim_critic1, input_dim_critic2, input_dim_critic3,
                 hidden_dim, which_agents, lr_critic=0.01, lr_policy=0.001):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            policy_type (int): form of policy actions
            num_in_critic (int): number of dimensions for critic input
        """
        
        # ---------------------------------------Policy Network---------------------------------------
        self.policyTypes = policyTypes
        self.eq_flag = eq_flag
        self.AwareOrNot = AwareOrNot
        self.policy = Debt_Policy_Networks_MAPPO(num_in_pol, policyTypes, eq_flag, AwareOrNot)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_policy)
        self.policy_loss = np.array([])

        # ---------------------------------------Critic Networks---------------------------------------
        self.critic = Critic_Networks(input_dim_critic1, input_dim_critic2, input_dim_critic3, hidden_dim, which_agents)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)
        self.critic_loss = {}
        self.critic_loss['Main'] = np.array([])

    def step(self, obs):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """

        action = self.policy(obs)
        
        return action                

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

#------------------------------Adaptation of MADDPG------------------------------

class AdmissionAgent_MADDPG(object):

    def __init__(self, num_in_pol, policyTypes, eq_flag, AwareOrNot,
                 input_dim_critic1, input_dim_critic2, input_dim_critic3,
                 hidden_dim, which_agents, lr_critic=0.01, lr_policy=0.001,
                 Fair_scheme = 0, N_steps = 1e7, decay_thresh=1e6, scale=0.015, sig_init=1,
                 noise_type='exponential', noise_init_type='uniform'):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            policy_type (int): form of policy actions
            num_in_critic (int): number of dimensions for critic input
        """

        # Policy Networks
        self.N_steps = N_steps
        self.decay_thresh = decay_thresh
        self.policyTypes = policyTypes
        self.eq_flag = eq_flag
        self.AwareOrNot = AwareOrNot
        self.policy = Admission_Policy_Networks_MADDPG(num_in_pol, policyTypes, eq_flag, AwareOrNot)
        self.target_policy = Admission_Policy_Networks_MADDPG(num_in_pol, policyTypes, eq_flag, AwareOrNot)
        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_policy)
        self.policy_loss = np.array([])
        self.scale = scale
        self.sig_init = sig_init
        self.noise_type = noise_type
        self.noise_init_type = noise_init_type

        # ---------------------------------------Critic Networks---------------------------------------
        self.critic = Critic_Networks(input_dim_critic1, input_dim_critic2, input_dim_critic3, hidden_dim, which_agents)


        # ---------------------------------------Target Critic Networks---------------------------------------
        self.target_critic = Critic_Networks(input_dim_critic1, input_dim_critic2, input_dim_critic3, hidden_dim, which_agents)
   

        # ---------------------------------------Optimizers---------------------------------------
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

        # --------------------------------------Q Loss function tracking---------------------------------------
        self.critic_loss = {}
        self.critic_loss['Main'] = np.array([])

    def step(self, obs, step_num, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        M,N = action.shape
        mu = np.zeros((M,N))
        sig = np.ones((M,N))*max([0.075*(100000-step_num)/100000, 0.02])
        noise = np.random.normal(mu,sig)
        action += Variable(Tensor(noise),
                                requires_grad=False)
        action = action.clamp(0, 1)
        
        return action


    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class DisburseAgent_MADDPG(object):

    def __init__(self, num_in_pol, policyTypes, eq_flag, AwareOrNot,
                 input_dim_critic1, input_dim_critic2, input_dim_critic3,
                 hidden_dim, which_agents, lr_critic=0.01, lr_policy=0.001,
                 Fair_scheme = 0, N_steps = 1e7, decay_thresh=1e6, scale=0.015, sig_init=1,
                 noise_type='exponential', noise_init_type='uniform'):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            policy_type (int): form of policy actions
            num_in_critic (int): number of dimensions for critic input
        """
        
        # Policy Network
        self.N_steps = N_steps
        self.decay_thresh = decay_thresh
        self.policyTypes = policyTypes
        self.eq_flag = eq_flag
        self.AwareOrNot = AwareOrNot
        self.policy = Disburse_Policy_Networks_MADDPG(num_in_pol, policyTypes, eq_flag, AwareOrNot)
        self.target_policy = Disburse_Policy_Networks_MADDPG(num_in_pol, policyTypes, eq_flag, AwareOrNot)
        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_policy)
        self.policy_loss = np.array([])
        self.scale = scale
        self.sig_init = sig_init
        self.noise_type = noise_type
        self.noise_init_type = noise_init_type

        # ---------------------------------------Critic Networks---------------------------------------
        self.critic = Critic_Networks(input_dim_critic1, input_dim_critic2, input_dim_critic3, hidden_dim, which_agents)

        # ---------------------------------------Target Critic Networks---------------------------------------
        self.target_critic = Critic_Networks(input_dim_critic1, input_dim_critic2, input_dim_critic3, hidden_dim, which_agents)

        # ---------------------------------------Update target critic---------------------------------------
        hard_update(self.target_critic, self.critic)

        # ---------------------------------------Optimizers---------------------------------------
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

        # --------------------------------------Q Loss function tracking---------------------------------------
        self.critic_loss = {}
        self.critic_loss['Main'] = np.array([])


    def step(self, obs, step_num, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        M,N = action.shape
        mu = np.zeros((M,N))
        sig = np.ones((M,N))*max([0.075*(100000-step_num)/100000, 0.02])
        noise = np.random.normal(mu,sig)
        action += Variable(Tensor(noise),
                                requires_grad=False)
        action = action.clamp(0, 1)
        
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

class DebtAgent_MADDPG(object):

    def __init__(self, num_in_pol, policyTypes, eq_flag, AwareOrNot,
                 input_dim_critic1, input_dim_critic2, input_dim_critic3,
                 hidden_dim, which_agents, lr_critic=0.01, lr_policy=0.001,
                 Fair_scheme=0, N_steps = 1e7, decay_thresh=1e6, scale=0.015, sig_init=1,
                 noise_type='exponential', noise_init_type='uniform'):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            policy_type (int): form of policy actions
            num_in_critic (int): number of dimensions for critic input
        """
        
        # Policy Network
        self.N_steps = N_steps
        self.decay_thresh = decay_thresh
        self.policyTypes = policyTypes
        self.eq_flag = eq_flag
        self.AwareOrNot = AwareOrNot
        self.policy = Debt_Policy_Networks_MADDPG(num_in_pol, policyTypes, eq_flag, AwareOrNot)
        self.target_policy = Debt_Policy_Networks_MADDPG(num_in_pol, policyTypes, eq_flag, AwareOrNot)
        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_policy)
        self.policy_loss = np.array([])
        self.scale = scale
        self.sig_init = sig_init
        self.noise_type = noise_type
        self.noise_init_type = noise_init_type

        # ---------------------------------------Critic Networks---------------------------------------
        self.critic = Critic_Networks(input_dim_critic1, input_dim_critic2, input_dim_critic3, hidden_dim, which_agents)

        # ---------------------------------------Target Critic Networks---------------------------------------
        self.target_critic = Critic_Networks(input_dim_critic1, input_dim_critic2, input_dim_critic3, hidden_dim, which_agents)

        # ---------------------------------------Update target critic---------------------------------------
        hard_update(self.target_critic, self.critic)

        # ---------------------------------------Optimizers---------------------------------------
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

        # --------------------------------------Q Loss function tracking---------------------------------------
        self.critic_loss = {}
        self.critic_loss['Main'] = np.array([])


    def scale_noise(self, scale):
        self.exploration.scale = scale

    def step(self, obs, step_num, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        M,N = action.shape
        mu = np.zeros((M,N))
        sig = np.ones((M,N))*max([0.075*(100000-step_num)/100000, 0.02])
        noise = np.random.normal(mu,sig)
        action += Variable(Tensor(noise),
                                requires_grad=False)
        action = action.clamp(0, 1)
        
        return action                

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])

########################################## NETWORK ARCHITECTURES ##########################################


class Critic_Networks(nn.Module):

    def __init__(self, input_dim_critic1, input_dim_critic2, input_dim_critic3, hidden_dim, which_agents, nonlin = F.leaky_relu):

        super(Critic_Networks, self).__init__()

        # Admission module layers (1 threshold for everyone)
        if input_dim_critic1 is not None:
            self.layer1_admit = nn.Linear(input_dim_critic1, input_dim_critic1)
            self.layer2_admit = nn.Linear(input_dim_critic1, input_dim_critic1)
            
            # Weight Initialization
            torch.nn.init.uniform_(self.layer1_admit.weight, -0.01, 0.01)
            torch.nn.init.uniform_(self.layer2_admit.weight, -0.01, 0.01)

        # Disbursement module layers (1 threshold for everyone)
        if input_dim_critic2 is not None:
            self.layer1_disburse = nn.Linear(input_dim_critic2, input_dim_critic2) 
            self.layer2_disburse = nn.Linear(input_dim_critic2, input_dim_critic2)  

            # Weight Initialization
            torch.nn.init.uniform_(self.layer1_disburse.weight, -0.01, 0.01)
            torch.nn.init.uniform_(self.layer2_disburse.weight, -0.01, 0.01)

        # Debt module layers (1 threshold for everyone)
        if input_dim_critic3 is not None:
            self.layer1_debt = nn.Linear(input_dim_critic3, input_dim_critic3)
            self.layer2_debt = nn.Linear(input_dim_critic3, input_dim_critic3)  

            # Weight Initialization
            torch.nn.init.uniform_(self.layer1_debt.weight, -0.01, 0.01)
            torch.nn.init.uniform_(self.layer2_debt.weight, -0.01, 0.01)

        # Post-merging layers
        input_dim_critic = 0                          
        if input_dim_critic1 is not None:             
            input_dim_critic += input_dim_critic1     
        if input_dim_critic2 is not None:             
            input_dim_critic += input_dim_critic2     
        if input_dim_critic3 is not None:             
            input_dim_critic += input_dim_critic3     
              
        self.layer3_merged = nn.Linear(input_dim_critic, 1)

        # Activation function        
        self.nonlin = nonlin

        # Final activation options
        self.sig = torch.sigmoid
        self.tanh = torch.tanh

        # Combine function
        self.F_comb = torch.sum

        # Weight Initialization
        torch.nn.init.uniform_(self.layer3_merged.weight, -0.01, 0.01)

    def forward(self, A, B, C):
        """
        Inputs:
            X (PyTorch Matrix): batch of observations (each row is an observation in the batch)

            Y: constant matrix for intermediate layer following batch permutation
        """
        torch.autograd.set_detect_anomaly(True)
        #--------Admissions structure of critic-----------
        #merge_layers = []
        if A is not None:

            # First layer
            A1 = self.layer1_admit(A)
            A1N = self.nonlin(A1)

            # Second layer
            A2 = self.layer2_admit(A1N)	
            A2N = self.nonlin(A2)		
        
            # Average across loan applicants
            X2 = self.F_comb(A2N,dim=0)

        #--------Disbursement structure of critic-----------
        if B is not None:

            # First layer
            B1 = self.layer1_disburse(B)
            B1N = self.nonlin(B1)

            # Second layer
            B2 = self.layer2_disburse(B1N)	    
            B2N = self.nonlin(B2)		
        
            # Average across loan applicants
            Y2 = self.F_comb(B2N,dim=0)	    

        #--------Debt structure of critic-----------
        if C is not None:

            # First layer
            C1 = self.layer1_debt(C)
            C1N = self.nonlin(C1)

            # Second layer
            C2 = self.layer2_debt(C1N) 	    
            C2N = self.nonlin(C2)		    
        
            # Average across loan applicants
            Z2 = self.F_comb(C2N,dim=0)	

        if A is not None and B is not None and C is not None:                                     
            merged = torch.cat((X2.unsqueeze(0),Y2.unsqueeze(0),Z2.unsqueeze(0)),dim=1)        
        elif A is None and B is not None and C is not None:                                       
            merged = torch.cat((Y2.unsqueeze(0),Z2.unsqueeze(0)),dim=1)                         
        elif A is not None and B is None and C is not None:                                       
            merged = torch.cat((X2.unsqueeze(0),Z2.unsqueeze(0)),dim=1)                         
        elif A is not None and B is not None and C is None:                                       
            merged = torch.cat((X2.unsqueeze(0),Y2.unsqueeze(0)),dim=1)                         

        output  = self.layer3_merged(merged) 

        return torch.squeeze(output)       

#-------------------MAPPO-------------------
class Admission_Policy_Networks_MAPPO(nn.Module):

    def __init__(self, input_dim, policyTypes, eq_flag, AwareOrNot, nonlin = F.leaky_relu):

        super(Admission_Policy_Networks_MAPPO, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, input_dim)

        # Admission threshold aware (2 thresholds, 1 per group)
        self.layer2_threshAware = nn.Linear(input_dim, input_dim)   # Multiplication by 2 becuase we need 2 parameters for sampling the gaussian for the action. ditto below.      
        self.layer3_threshAware = nn.Linear(input_dim,1*4)

        # Admission threshold unaware (1 threshold for everyone)
        self.layer2_threshUnaware = nn.Linear(input_dim, input_dim)
        self.layer3_threshUnaware = nn.Linear(input_dim,1*2)

        # Admission scores (score per individual)
        self.layer2_score = nn.Linear(input_dim, input_dim)
        self.layer3_score = nn.Linear(input_dim,1*2)

        # Activation function        
        self.nonlin = nonlin

        # Final activation
        self.sig = torch.sigmoid

        # Policy Type
        self.policyTypes = policyTypes

        # Equal admissions rate
        self.eq_flag = eq_flag

        # Awareness of sensitive attribute or not
        self.AwareOrNot = AwareOrNot

        # To help remove nans
        self.zeros_threshAware = torch.zeros(0,input_dim)
        self.zeros_threshUnaware = torch.zeros(0,input_dim)

        # Combine function
        #self.F_comb = torch.mean
        self.F_comb = torch.sum

        self.eps = 1e-7    
        
        # Initialize weights
        torch.nn.init.uniform_(self.layer1.weight, -0.01, 0.01)			
        torch.nn.init.uniform_(self.layer2_threshAware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer3_threshAware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer2_threshUnaware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer3_threshUnaware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer2_score.weight, -0.01, 0.01)		
        torch.nn.init.uniform_(self.layer3_score.weight, -0.01, 0.01)		

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): batch of observations (each row is an observation in the batch)

            Y: constant matrix for intermediate layer following batch permutation
        """

        # First layer run through
        X1 = self.layer1(X) 
        X1N = self.nonlin(X1)

        if self.policyTypes == 1 and not self.eq_flag and self.AwareOrNot:  # Threshold policy (one per group)

            # Second layer run through
            X2 = self.layer2_threshAware(X1N)		
            X2N = self.nonlin(X2)		    

            # Average across loan applicants
            if X2N.shape[0]==0:							    
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0) + self.zeros_threshAware	    
            else:								    
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0)			

            # Next layer
            X3 = self.layer3_threshAware(Z2)				    
            X3N = self.sig(X3)
        
        elif self.policyTypes == 1:       # Threshold policy (one tptal)
            # Second layer run through
            X2 = self.layer2_threshUnaware(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            if X2N.shape[0]==0:							    
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0) + self.zeros_threshUnaware 
            else:									
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0)			    

            # Next layer
            X3 = self.layer3_threshUnaware(Z2)					
            X3N = self.sig(X3)

        else:                                    # Score policy
            # Second layer run through
            X2 = self.layer2_score(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            Z2 = X2N-torch.mean(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_score(Z2)					      
            X3N = self.sig(X3)

        # Returns
        M,N = X3N.shape
        sig_idx = int(N/2)
        sig = torch.ones(M,sig_idx)*X3N[:,sig_idx:]/10
        dist = Normal(X3N[:,:sig_idx], sig)

        act = dist.sample()
        log_prob = dist.log_prob(act).sum()
        entropy = dist.entropy().sum()
            
        return X3N, act, log_prob, entropy             # mu+sig, sample, log prob, entropy of sample

    def get_fake_actions(self, torch_agent_true_action):

        ''' Fake actions only are required when the obs is empty. Otherwise,
        torch_agent_true_actions = torch_agent_fake_actions'''
        
        torch_agent_true_action = torch_agent_true_action.detach().numpy()
            
        MM,NN = torch_agent_true_action.shape
            
        act_4_train = np.zeros((1,NN)) # Action of all zeros if no observations provided
            
        torch_agent_fake_action = Variable(torch.Tensor(act_4_train),requires_grad=False)

            
        return torch_agent_fake_action


##### THIS AGENT IS NOT READY FOR INTEGRATION YET ######
class Disburse_Policy_Networks_MAPPO(nn.Module):

    def __init__(self, input_dim, policyTypes, eq_flag, AwareOrNot, nonlin = F.leaky_relu):

        super(Disburse_Policy_Networks_MAPPO, self).__init__()

        self.layer1 = nn.Linear(input_dim, 1)

        # Insurance Premiums
        self.layer2_score = nn.Linear(input_dim+1, 2)

        # Activation function        
        self.nonlin = nonlin

        # Final activation
        self.sig = torch.sigmoid

        # To help remove nans
        self.zeros = torch.zeros(0, 1)

        # Policy Type
        self.policyTypes = policyTypes

        self.eps = 1e-9
        

    def forward(self, X, return_log_pi=False, sample=False):
        """
        Inputs:
            X (PyTorch Matrix): batch of observations (each row is an observation in the batch)

            Y: constant matrix for intermediate layer following batch permutation
        """

        # First layer run through
        M,N = X.shape
        X1 = self.layer1(X)
        X1N = self.nonlin(X1)
        
        # Average across loan applicants
        if X1N.shape[0]==0:
            Z2 = torch.multiply(torch.ones(M,1), torch.mean(X1N,dim=0).unsqueeze(0) + self.zeros)
        else:
            Z2 = torch.multiply(torch.ones(M,1), torch.mean(X1N,dim=0).unsqueeze(0) )

        Z3 = torch.hstack((X,Z2))
        X3 = self.layer2_score(Z3)
        X3N = self.sig(X3)

        # Returns
        dist = Normal(X3N[:,:1], X3N[:,1:2]/10)

        act = dist.sample()
        log_prob = dist.log_prob(act).sum()
        entropy = dist.entropy().sum()
            
        return X3N, act, log_prob, entropy             # mu+sig, sample, log prob, entropy of sample  

class Debt_Policy_Networks_MAPPO(nn.Module):

    def __init__(self, input_dim, policyTypes, eq_flag, AwareOrNot, nonlin = F.leaky_relu):

        super(Debt_Policy_Networks_MAPPO, self).__init__()

        self.layer1 = nn.Linear(input_dim, input_dim)

        # Debt forgiveness percentage aware (2 thresholds, 1 per group)
        self.layer2_globalAware = nn.Linear(input_dim, input_dim)
        self.layer3_globalAware = nn.Linear(input_dim, 1*4)

        # Debt forgiveness percentage unaware (1 threshold for everyone)
        self.layer2_globalUnaware = nn.Linear(input_dim, input_dim)
        self.layer3_globalUnaware = nn.Linear(input_dim, 1*2)

        # Debt forgiveness percentage (per individual)
        self.layer2_individual = nn.Linear(input_dim, input_dim)
        self.layer3_individual = nn.Linear(input_dim, 1*2)

        # Activation function        
        self.nonlin = nonlin

        # Final activation
        self.sig = torch.sigmoid

        # Policy Type
        self.policyTypes = policyTypes

        # Equal debt forgiveness rate
        self.eq_flag = eq_flag

        # Awareness of sensitive attribute or not
        self.AwareOrNot = AwareOrNot

        # To help remove nans
        self.zeros_globalAware = torch.zeros(0,input_dim)
        self.zeros_globalUnaware = torch.zeros(0,input_dim)

        # Combine function
        self.F_comb = torch.sum

        self.eps = 1e-9

        # Initialize weights
        torch.nn.init.uniform_(self.layer1.weight, -0.01, 0.01)		    
        torch.nn.init.uniform_(self.layer2_globalAware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer3_globalAware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer2_globalUnaware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer3_globalUnaware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer2_individual.weight, -0.01, 0.01)		
        torch.nn.init.uniform_(self.layer3_individual.weight, -0.01, 0.01)		

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): batch of observations (each row is an observation in the batch)

            Y: constant matrix for intermediate layer following batch permutation
        """

        # First layer run through
        X1 = self.layer1(X)
        X1N = self.nonlin(X1)

        if self.policyTypes == 2 and not self.eq_flag and self.AwareOrNot:       # Threshold policy (one total)

            # Second layer run through
            X2 = self.layer2_globalAware(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            if X2N.shape[0]==0:
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0) + self.zeros_globalAware
            else:
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_globalAware(Z2)
            X3N = self.sig(X3)

        elif self.policyTypes == 2:       # Threshold policy (one total)

            # Second layer run through
            X2 = self.layer2_globalUnaware(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            if X2N.shape[0]==0:
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0) + self.zeros_globalUnaware
            else:
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_globalUnaware(Z2)
            X3N = self.sig(X3)

        else:

            # Second layer run through
            X2 = self.layer2_individual(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            Z2 = X2N-torch.mean(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_individual(Z2)
            X3N = self.sig(X3)


        # Returns
        M,N = X3N.shape
        sig_idx = int(N/2)
        sig = torch.ones(M,sig_idx)*X3N[:,sig_idx:]/10
        dist = Normal(X3N[:,:sig_idx], sig)

        act = dist.sample()
        log_prob = dist.log_prob(act).sum()
        entropy = dist.entropy().sum()
            
        return X3N, act, log_prob, entropy             # mu+sig, sample, log prob, entropy of sample        
 
    def get_fake_actions(self, torch_agent_true_action):

        ''' Fake actions only are required when the obs is empty. Otherwise,
        torch_agent_true_actions = torch_agent_fake_actions'''
        
        torch_agent_true_action = torch_agent_true_action.detach().numpy()
            
        MM,NN = torch_agent_true_action.shape
            
        act_4_train = np.zeros((1,NN)) # Action of all zeros if no observations provided
            
        torch_agent_fake_action = Variable(torch.Tensor(act_4_train),requires_grad=False)

            
        return torch_agent_fake_action

#--------------------------------MADDPG--------------------------------
class Admission_Policy_Networks_MADDPG(nn.Module):

    def __init__(self, input_dim, policyTypes, eq_flag, AwareOrNot, nonlin = F.leaky_relu):

        super(Admission_Policy_Networks_MADDPG, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, input_dim)

        # Admission threshold aware (2 thresholds, 1 per group)
        self.layer2_threshAware = nn.Linear(input_dim, input_dim)   # Multiplication by 2 becuase we need 2 parameters for sampling the gaussian for the action. ditto below.      
        self.layer3_threshAware = nn.Linear(input_dim,1*2)

        # Admission threshold unaware (1 threshold for everyone)
        self.layer2_threshUnaware = nn.Linear(input_dim, input_dim)
        self.layer3_threshUnaware = nn.Linear(input_dim,1*1)

        # Admission scores (score per individual)
        self.layer2_score = nn.Linear(input_dim, input_dim)
        self.layer3_score = nn.Linear(input_dim,1*1)

        # Activation function        
        self.nonlin = nonlin

        # Final activation
        self.sig = torch.sigmoid

        # Policy Type
        self.policyTypes = policyTypes

        # Equal admissions rate
        self.eq_flag = eq_flag

        # Awareness of sensitive attribute or not
        self.AwareOrNot = AwareOrNot

        # To help remove nans
        self.zeros_threshAware = torch.zeros(0,input_dim)
        self.zeros_threshUnaware = torch.zeros(0,input_dim)

        # Combine function
        self.F_comb = torch.sum

        self.eps = 1e-9

        # Initialize weights
        torch.nn.init.uniform_(self.layer1.weight, -0.01, 0.01)			
        torch.nn.init.uniform_(self.layer2_threshAware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer3_threshAware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer2_threshUnaware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer3_threshUnaware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer2_score.weight, -0.01, 0.01)		
        torch.nn.init.uniform_(self.layer3_score.weight, -0.01, 0.01)	
        

    def forward(self, X, return_log_pi=False, sample=False):
        """
        Inputs:
            X (PyTorch Matrix): batch of observations (each row is an observation in the batch)

            Y: constant matrix for intermediate layer following batch permutation
        """

        # First layer run through
        X1 = self.layer1(X)
        X1N = self.nonlin(X1)

        if self.policyTypes == 1 and not self.eq_flag and self.AwareOrNot:  # Threshold policy (one per group)

            # Second layer run through
            X2 = self.layer2_threshAware(X1N)
            X2N = self.nonlin(X2)

            # Average across loan applicants
            if X2N.shape[0]==0:
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0) + self.zeros_threshAware
            else:
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_threshAware(Z2)
            X3N = self.sig(X3)
        
        elif self.policyTypes == 1:       # Threshold policy (one tptal)
            # Second layer run through
            X2 = self.layer2_threshUnaware(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            if X2N.shape[0]==0:
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0) + self.zeros_threshUnaware
            else:
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_threshUnaware(Z2)
            X3N = self.sig(X3)

        else:                                    # Score policy
            # Second layer run through
            X2 = self.layer2_score(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            Z2 = X2N-torch.mean(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_score(Z2)
            X3N = self.sig(X3)


        # Returns
        if not sample and not return_log_pi:
            rets = X3N
        else:
            rets = [X3N]

        if sample:
            action = self.policy(obs)
            M,N = action.shape
        
            mu = np.zeros((M,N))
            sig = np.ones((M,N))*0.05
            noise = np.random.normal(mu,sig)
            action += Variable(Tensor(noise),
                                requires_grad=False)

            action = action.clamp(0, 1)
            rets.append(action)

        if return_log_pi:
            # Return entropy of selected action
            sig_idx = int(X3N.shape[1]/2)
            sigs = torch.flatten(X3N[:,sig_idx:])
            entropy = 0.5*torch.log(torch.prod( sigs )+ self.eps)+(len(sigs)/2)*(1+np.log(2*np.pi)) # Entropy: Actions sampled from Gaussian
            #rets = [rets]
            rets.append(entropy)

            # Log of policy action 
            log_rets = torch.log(X3N)
            rets.append(log_rets)
            
        return rets

class Disburse_Policy_Networks_MADDPG(nn.Module):

    def __init__(self, input_dim, policyTypes, eq_flag, AwareOrNot, nonlin = F.leaky_relu):

        super(Disburse_Policy_Networks_MADDPG, self).__init__()

        self.layer1 = nn.Linear(input_dim, 1)

        # Hospital queue
        self.layer2_score = nn.Linear(input_dim+1, 1)

        # Activation function        
        self.nonlin = nonlin

        # Final activation
        self.sig = torch.sigmoid

        # To help remove nans
        self.zeros = torch.zeros(0, 1)


        # Policy Type
        self.policyTypes = policyTypes

        self.eps = 1e-9
        

    def forward(self, X, return_log_pi=False, sample=False):
        """
        Inputs:
            X (PyTorch Matrix): batch of observations (each row is an observation in the batch)

            Y: constant matrix for intermediate layer following batch permutation
        """

        # First layer run through
        M,N = X.shape
        X1 = self.layer1(X)
        X1N = self.nonlin(X1)
        
        # Average across loan applicants
        if X1N.shape[0]==0:
            Z2 = torch.multiply(torch.ones(M,1), torch.mean(X1N,dim=0).unsqueeze(0) + self.zeros)
        else:
            Z2 = torch.multiply(torch.ones(M,1), torch.mean(X1N,dim=0).unsqueeze(0) )
            
        Z3 = torch.hstack((X,Z2))
        X3 = self.layer2_score(Z3)
        X3N = self.sig(X3)

        # Returns
        if not sample and not return_log_pi:
            rets = X3N
        else:
            rets = [X3N]

        if sample:
            action = self.policy(obs)
            M,N = action.shape
        
            mu = np.zeros((M,N))
            sig = np.ones((M,N))*0.05
            noise = np.random.normal(mu,sig)
            action += Variable(Tensor(noise),
                                requires_grad=False)

            action = action.clamp(0, 1)
            rets.append(action)

        if return_log_pi:
            # Return entropy of selected action
            sig_idx = int(X3N.shape[1]/2)
            sigs = torch.flatten(X3N[:,sig_idx:])
            entropy = 0.5*torch.log(torch.prod( sigs )+ self.eps)+(len(sigs)/2)*(1+np.log(2*np.pi)) # Entropy: Actions sampled from Gaussian
            #rets = [rets]
            rets.append(entropy)

            # Log of policy action 
            log_rets = torch.log(X3N)
            rets.append(log_rets)

        return rets

class Debt_Policy_Networks_MADDPG(nn.Module):

    def __init__(self, input_dim, policyTypes, eq_flag, AwareOrNot, nonlin = F.leaky_relu):

        super(Debt_Policy_Networks_MADDPG, self).__init__()

        self.layer1 = nn.Linear(input_dim, input_dim)

        # Debt forgiveness percentage aware (2 thresholds, 1 per group)
        self.layer2_globalAware = nn.Linear(input_dim, input_dim)
        self.layer3_globalAware = nn.Linear(input_dim, 1*2)

        # Debt forgiveness percentage unaware (1 threshold for everyone)
        self.layer2_globalUnaware = nn.Linear(input_dim, input_dim)
        self.layer3_globalUnaware = nn.Linear(input_dim, 1*1)

        # Debt forgiveness percentage (per individual)
        self.layer2_individual = nn.Linear(input_dim, input_dim)
        self.layer3_individual = nn.Linear(input_dim, 1*1)

        # Activation function        
        self.nonlin = nonlin

        # Final activation
        self.sig = torch.sigmoid

        # Policy Type
        self.policyTypes = policyTypes

        # Equal debt forgiveness rate
        self.eq_flag = eq_flag

        # Awareness of sensitive attribute or not
        self.AwareOrNot = AwareOrNot

        # To help remove nans
        self.zeros_globalAware = torch.zeros(0,input_dim)
        self.zeros_globalUnaware = torch.zeros(0,input_dim)

        # Combine function
        self.F_comb = torch.sum

        self.eps = 1e-9

        # Initialize weights
        torch.nn.init.uniform_(self.layer1.weight, -0.01, 0.01)		    
        torch.nn.init.uniform_(self.layer2_globalAware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer3_globalAware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer2_globalUnaware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer3_globalUnaware.weight, -0.01, 0.01)	
        torch.nn.init.uniform_(self.layer2_individual.weight, -0.01, 0.01)		
        torch.nn.init.uniform_(self.layer3_individual.weight, -0.01, 0.01)

    def forward(self, X, return_log_pi=False, sample=False):
        """
        Inputs:
            X (PyTorch Matrix): batch of observations (each row is an observation in the batch)

            Y: constant matrix for intermediate layer following batch permutation
        """

        # First layer run through
        X1 = self.layer1(X)
        X1N = self.nonlin(X1)

        if self.policyTypes == 2 and not self.eq_flag and self.AwareOrNot:       # Threshold policy (one total)

            # Second layer run through
            X2 = self.layer2_globalAware(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            if X2N.shape[0]==0:
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0) + self.zeros_globalAware
            else:
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_globalAware(Z2)
            X3N = self.sig(X3)

        elif self.policyTypes == 2:       # Threshold policy (one total)

            # Second layer run through
            X2 = self.layer2_globalUnaware(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            if X2N.shape[0]==0:
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0) + self.zeros_globalUnaware
            else:
                Z2 = self.F_comb(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_globalUnaware(Z2)
            X3N = self.sig(X3)

        else:

            # Second layer run through
            X2 = self.layer2_individual(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            Z2 = X2N-torch.mean(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_individual(Z2)
            X3N = self.sig(X3)


        # Returns
        if not sample and not return_log_pi:
            rets = X3N
        else:
            rets = [X3N]

        if sample:
            action = self.policy(obs)
            M,N = action.shape
        
            mu = np.zeros((M,N))
            sig = np.ones((M,N))*0.05
            noise = np.random.normal(mu,sig)
            action += Variable(Tensor(noise),
                                requires_grad=False)

            action = action.clamp(0, 1)
            rets.append(action)

        if return_log_pi:
            # Return entropy of selected action
            sig_idx = int(X3N.shape[1]/2)
            sigs = torch.flatten(X3N[:,sig_idx:])
            entropy = 0.5*torch.log(torch.prod( sigs )+ self.eps)+(len(sigs)/2)*(1+np.log(2*np.pi)) # Entropy: Actions sampled from Gaussian
            rets.append(entropy)

            # Log of policy action 
            log_rets = torch.log(X3N)
            rets.append(log_rets)

        return rets

#--------------------------------MAAC---------------------------------
class Admission_Policy_Networks_MAAC(nn.Module):

    def __init__(self, input_dim, policyTypes, eq_flag, AwareOrNot, nonlin = F.relu):

        super(Admission_Policy_Networks_MAAC, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, input_dim)

        # Admission threshold aware (2 thresholds, 1 per group)
        self.layer2_threshAware = nn.Linear(input_dim, 1*2)   # Multiplication by 2 becuase we need 2 parameters for sampling the gaussian for the action. ditto below.      
        self.layer3_threshAware = nn.Linear(1*2,1*2)

        # Admission threshold unaware (1 threshold for everyone)
        self.layer2_threshUnaware = nn.Linear(input_dim, 1*1)
        self.layer3_threshUnaware = nn.Linear(1*1,1*1)

        # Admission scores (score per individual)
        self.layer2_score = nn.Linear(input_dim, 1*1)
        self.layer3_score = nn.Linear(1*1,1*1)

        # Activation function        
        self.nonlin = nonlin

        # Final activation
        self.sig = torch.sigmoid

        # Policy Type
        self.policyTypes = policyTypes

        # Equal admissions rate
        self.eq_flag = eq_flag

        # Awareness of sensitive attribute or not
        self.AwareOrNot = AwareOrNot

        # To help remove nans
        self.zeros_threshAware = torch.zeros(0,1*2)
        self.zeros_threshUnaware = torch.zeros(0,1*1)

        self.eps = 1e-9
        

    def forward(self, X, return_log_pi=False, test=False, policy_update=False, N_base_samp=30, empty_obs=False):
        """
        Inputs:
            X (PyTorch Matrix): batch of observations (each row is an observation in the batch)

            Y: constant matrix for intermediate layer following batch permutation
        """

        # First layer run through
        X1 = self.layer1(X)
        X1N = self.nonlin(X1)


        if self.policyTypes == 1 and not self.eq_flag and self.AwareOrNot:  # Threshold policy (one per group)

            # Second layer run through
            X2 = self.layer2_threshAware(X1N)
            X2N = self.nonlin(X2)

            # Average across loan applicants
            if X2N.shape[0]==0:
                Z2 = torch.mean(X2N,dim=0).unsqueeze(0) + self.zeros_threshAware
            else:
                Z2 = torch.mean(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_threshAware(Z2)
            X3N = self.sig(X3)
        
        elif self.policyTypes == 1:       # Threshold policy (one tptal)
            # Second layer run through
            X2 = self.layer2_threshUnaware(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            if X2N.shape[0]==0:
                Z2 = torch.mean(X2N,dim=0).unsqueeze(0) + self.zeros_threshUnaware
            else:
                Z2 = torch.mean(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_threshUnaware(Z2)
            X3N = self.sig(X3)

        else:                                    # Score policy
            # Second layer run through
            X2 = self.layer2_score(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            Z2 = X2N-torch.mean(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_score(Z2)
            X3N = self.sig(X3)

        # Returns
        if test:
            rets = X3N

        else:
            
            M,N = X3N.shape
            sig = Tensor(np.ones((M,N))*0.05).detach()
            dist = Normal(X3N, sig)
            
            if not policy_update:
                act = dist.sample()             # action
                rets = act
                if return_log_pi:
                    log_prob = torch.sum(dist.log_prob(act),dim=1).reshape(M,1) # log of joint prob. Recall: log(ab) = log(a) + log(b)
                    rets = [rets]
                    rets.append(log_prob)
            else:
                acts = [dist.sample() for _ in range(N_base_samp)]
                act = dist.sample()
                log_prob = torch.sum(dist.log_prob(act),dim=1).reshape(M,1) # log of joint prob. Recall: log(ab) = log(a) + log(b)
                # For dealing with empty obs and actions
                if empty_obs:
                    acts = [ self.get_fake_actions(acts[nn]) for nn in range(N_base_samp) ]
                    act = self.get_fake_actions(act)
                rets = [act, log_prob, acts]
            
        return rets

    def get_fake_actions(self, torch_agent_true_action):

        ''' Fake actions only are required when the obs is empty. Otherwise,
        torch_agent_true_actions = torch_agent_fake_actions'''
        
        torch_agent_true_action = torch_agent_true_action.detach().numpy()
            
        MM,NN = torch_agent_true_action.shape
            
        act_4_train = np.zeros((1,NN)) # Action of all zeros if no observations provided
            
        torch_agent_fake_action = Variable(torch.Tensor(act_4_train),requires_grad=False)

            
        return torch_agent_fake_action

class Disburse_Policy_Networks_MAAC(nn.Module):

    def __init__(self, input_dim, policyTypes, eq_flag, AwareOrNot, nonlin = F.relu):

        super(Disburse_Policy_Networks_MAAC, self).__init__()

        self.layer1 = nn.Linear(input_dim, 1)

        # Hospital queue
        self.layer2_score = nn.Linear(input_dim+1, 1)

        # Activation function        
        self.nonlin = nonlin

        # Final activation
        self.sig = torch.sigmoid

        # To help remove nans
        self.zeros = torch.zeros(0, 1)

        # Policy Type
        self.policyTypes = policyTypes

        self.eps = 1e-9
        

    def forward(self, X, return_log_pi=False, test=False, policy_update=False, N_base_samp=30, empty_obs=False):
        """
        Inputs:
            X (PyTorch Matrix): batch of observations (each row is an observation in the batch)

            Y: constant matrix for intermediate layer following batch permutation
        """

        # First layer run through
        M,N = X.shape
        X1 = self.layer1(X)
        X1N = self.nonlin(X1)
        
        # Average across loan applicants
        if X1N.shape[0]==0:
            Z2 = torch.multiply(torch.ones(M,1), torch.mean(X1N,dim=0).unsqueeze(0) + self.zeros)
        else:
            Z2 = torch.multiply(torch.ones(M,1), torch.mean(X1N,dim=0).unsqueeze(0) )

        Z3 = torch.hstack((X,Z2))
        X3 = self.layer2_score(Z3)
        X3N = self.sig(X3)

        # Returns
        if test:
            rets = X3N
        else:
            
            M,N = X3N.shape
            sig = Tensor(np.ones((M,N))*0.05).detach()
            dist = Normal(X3N, sig)
        
            if not policy_update:
                act = dist.sample()             # action
                rets = act
                if return_log_pi:
                    log_prob = torch.sum(dist.log_prob(act),dim=1).reshape(M,1) # log of joint prob. Recall: log(ab) = log(a) + log(b)
                    rets = [rets]
                    rets.append(log_prob)
            else:
                acts = [dist.sample() for _ in range(N_base_samp)]
                act = dist.sample()
                log_prob = torch.sum(dist.log_prob(act),dim=1).reshape(M,1) # log of joint prob. Recall: log(ab) = log(a) + log(b)
                
                # For dealing with empty obs and actions
                if empty_obs:
                    acts = [ self.get_fake_actions(acts[nn]) for nn in range(N_base_samp) ]
                    act = self.get_fake_actions(act)
                    
                rets = [act, log_prob, acts]
            
        return rets

    def get_fake_actions(self, torch_agent_true_action):

        ''' Fake actions only are required when the obs is empty. Otherwise,
        torch_agent_true_actions = torch_agent_fake_actions'''
        
        torch_agent_true_action = torch_agent_true_action.detach().numpy()
            
        MM,NN = torch_agent_true_action.shape
            
        act_4_train = np.zeros((1,NN)) # Action of all zeros if no observations provided
            
        torch_agent_fake_action = Variable(torch.Tensor(act_4_train),requires_grad=False)

            
        return torch_agent_fake_action


class Debt_Policy_Networks_MAAC(nn.Module):

    def __init__(self, input_dim, policyTypes, eq_flag, AwareOrNot, nonlin = F.relu):

        super(Debt_Policy_Networks_MAAC, self).__init__()

        self.layer1 = nn.Linear(input_dim, input_dim)

        # Debt forgiveness percentage aware (2 thresholds, 1 per group)
        self.layer2_globalAware = nn.Linear(input_dim, 1*2)
        self.layer3_globalAware = nn.Linear(1*2, 1*2)

        # Debt forgiveness percentage unaware (1 threshold for everyone)
        self.layer2_globalUnaware = nn.Linear(input_dim, 1*1)
        self.layer3_globalUnaware = nn.Linear(1*1, 1*1)

        # Debt forgiveness percentage (per individual)
        self.layer2_individual = nn.Linear(input_dim, 1*1)
        self.layer3_individual = nn.Linear(1*1, 1*1)

        # Activation function        
        self.nonlin = nonlin

        # Final activation
        self.sig = torch.sigmoid

        # Policy Type
        self.policyTypes = policyTypes

        # Equal debt forgiveness rate
        self.eq_flag = eq_flag

        # Awareness of sensitive attribute or not
        self.AwareOrNot = AwareOrNot

        # To help remove nans
        self.zeros_globalAware = torch.zeros(0,1*2)
        self.zeros_globaUnaware = torch.zeros(0,1*1)

        self.eps = 1e-9

    def forward(self, X, return_log_pi=False, test=False, policy_update=False, N_base_samp=30, empty_obs=False):
        """
        Inputs:
            X (PyTorch Matrix): batch of observations (each row is an observation in the batch)

            Y: constant matrix for intermediate layer following batch permutation
        """

        # First layer run through
        X1 = self.layer1(X)
        X1N = self.nonlin(X1)

        if self.policyTypes == 2 and not self.eq_flag and self.AwareOrNot:       # Threshold policy (one total)

            # Second layer run through
            X2 = self.layer2_globalAware(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            if X2N.shape[0]==0:
                Z2 = torch.mean(X2N,dim=0).unsqueeze(0) + self.zeros_globalAware
            else:
                Z2 = torch.mean(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_globalAware(Z2)
            X3N = self.sig(X3)

        elif self.policyTypes == 2:       # Threshold policy (one total)

            # Second layer run through
            X2 = self.layer2_globalUnaware(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            if X2N.shape[0]==0:
                Z2 = torch.mean(X2N,dim=0).unsqueeze(0) + self.zeros_globalUnaware
            else:
                Z2 = torch.mean(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_globalUnaware(Z2)
            X3N = self.sig(X3)

        else:

            # Second layer run through
            X2 = self.layer2_individual(X1N)
            X2N = self.nonlin(X2)
        
            # Average across loan applicants
            Z2 = X2N-torch.mean(X2N,dim=0).unsqueeze(0)

            # Next layer
            X3 = self.layer3_individual(Z2)
            X3N = self.sig(X3)

        # Returns

        if test:
            rets = X3N

        else:
            
            M,N = X3N.shape
            sig = Tensor(np.ones((M,N))*0.05).detach()
            dist = Normal(X3N, sig)
        
            if not policy_update:
                act = dist.sample()             # action
                rets = act
                if return_log_pi:
                    log_prob = torch.sum(dist.log_prob(act),dim=1).reshape(M,1) # log of joint prob. Recall: log(ab) = log(a) + log(b)
                    rets = [rets]
                    rets.append(log_prob)
            else:
                acts = [dist.sample() for _ in range(N_base_samp)]
                act = dist.sample()
                log_prob = torch.sum(dist.log_prob(act),dim=1).reshape(M,1) # log of joint prob. Recall: log(ab) = log(a) + log(b)

                # For dealing with empty obs and actions
                if empty_obs:
                    acts = [ self.get_fake_actions(acts[nn]) for nn in range(N_base_samp) ]
                    act = self.get_fake_actions(act)
                rets = [act, log_prob, acts]
            
        return rets

    def get_fake_actions(self, torch_agent_true_action):

        ''' Fake actions only are required when the obs is empty. Otherwise,
        torch_agent_true_actions = torch_agent_fake_actions'''
        
        torch_agent_true_action = torch_agent_true_action.detach().numpy()
            
        MM,NN = torch_agent_true_action.shape
            
        act_4_train = np.zeros((1,NN)) # Action of all zeros if no observations provided
            
        torch_agent_fake_action = Variable(torch.Tensor(act_4_train),requires_grad=False)

            
        return torch_agent_fake_action
