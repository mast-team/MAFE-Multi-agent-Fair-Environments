import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
import pdb


class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        if agents is None:
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets





###### GET INPUT DIMENSIONS STAIGHT FOR CRITICS
class LoanAttentionCritic_BACKUP(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
#    def __init__(self, input_dim_critic1, input_dim_critic2, input_dim_critic3,
#                 input_dim_pol1, input_dim_pol2, input_dim_pol3,
#                 which_agents, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
    def __init__(self, input_dim_critic1, input_dim_critic2, input_dim_critic3,
                 input_dim_pol1, input_dim_pol2, input_dim_pol3,
                 which_agents, odim= 1, hidden_dim=32, norm_in=True, attend_heads=1):

        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(LoanAttentionCritic, self).__init__()

        assert (hidden_dim % attend_heads) == 0
        #self.sa_sizes = sa_sizes
        #self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.state_encoders = nn.ModuleList()
        self.critic_losses = []

        # Admission module layers (1 threshold for everyone)
        #self.layer1_admit = nn.Linear(input_dim_critic1, hidden_dim)
        
        if input_dim_critic1 is not None:

            encoder = nn.Sequential()
            encoder.add_module('enc_fc1', nn.Linear(input_dim_critic1, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            encoder.add_module('enc_m1', MEAN())
            self.critic_encoders.append(encoder)
            #### ADAPT THIS TO MY APPLICATION########
            
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim)) ##FIX DIMENSIONS LATER!!!!!!!!
            self.critics.append(critic)
            self.critic_losses.append(np.array([]))
            ####----------------------------########
            
            state_encoder = nn.Sequential()
            state_encoder.add_module('s_enc_fc1', nn.Linear(input_dim_pol1, hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            state_encoder.add_module('s_enc_m1', MEAN())
            self.state_encoders.append(state_encoder)
        else:
            self.critic_encoders.append(None)
            self.critics.append(None)
            self.critic_losses.append(None)
            self.state_encoders.append(None)

        if input_dim_critic2 is not None:
            # Disbursement module layers (1 threshold for everyone)
            #self.layer1_disburse = nn.Linear(input_dim_critic2, hidden_dim)
            encoder = nn.Sequential()
            encoder.add_module('enc_fc1', nn.Linear(input_dim_critic2, hidden_dim))
            encoder.add_module('enc_n1', nn.LeakyReLU())
            encoder.add_module('enc_m1', MEAN())
            self.critic_encoders.append(encoder)
            #### ADAPT THIS TO MY APPLICATION########
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)
            self.critic_losses.append(np.array([]))
            ####----------------------------########
            state_encoder = nn.Sequential()
            state_encoder.add_module('s_enc_fc1', nn.Linear(input_dim_pol2, hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            state_encoder.add_module('s_enc_m1', MEAN())
            self.state_encoders.append(state_encoder)
        else:
            self.critic_encoders.append(None)
            self.critics.append(None)
            self.critic_losses.append(None)
            self.state_encoders.append(None)

        if input_dim_critic3 is not None:
            # Debt module layers (1 threshold for everyone)
            #self.layer1_debt = nn.Linear(input_dim_critic3, hidden_dim)
            encoder = nn.Sequential()
            encoder.add_module('enc_fc1', nn.Linear(input_dim_critic3, hidden_dim))
            encoder.add_module('enc_n1', nn.LeakyReLU())
            encoder.add_module('enc_m1', MEAN())
            self.critic_encoders.append(encoder)
            #### ADAPT THIS TO MY APPLICATION########
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)
            self.critic_losses.append(np.array([]))
            ####----------------------------########
            state_encoder = nn.Sequential()
            state_encoder.add_module('s_enc_fc1', nn.Linear(input_dim_pol3, hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            state_encoder.add_module('s_enc_m1', MEAN())
            self.state_encoders.append(state_encoder)
        else:
            self.critic_encoders.append(None)
            self.critics.append(None)
            self.critic_losses.append(None)
            self.state_encoders.append(None)

        # Merge module layers
        self.which_agents = which_agents
        self.N_agents = 3#sum(which_agents)

        # Activation function        
        #self.nonlin = nonlin

        # iterate over agents
        #for sdim, adim in sa_sizes:
            #idim = sdim + adim
            #odim = adim
            #encoder = nn.Sequential()
            #if norm_in:
            #    encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
            #                                                affine=False))
            #encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            #encoder.add_module('enc_nl', nn.LeakyReLU())
            #self.critic_encoders.append(encoder)
            #critic = nn.Sequential()
            #critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
            #                                          hidden_dim))
            #critic.add_module('critic_nl', nn.LeakyReLU())
            #critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            #self.critics.append(critic)

            #state_encoder = nn.Sequential()
            #if norm_in:
            #    state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
            #                                sdim, affine=False))
            #state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
            #                                                hidden_dim))
            #state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            #self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

        ### EVERYTHING UP TO THIS POINT IN THE INIT SHOULD BE CONCEPTUALLY OKAY. STILL NEED TO TEST THOUGH, BUT START CODING NEXT PART FIRST####

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.N_agents)

    def forward(self, states, actions, inps, agents=None, return_q=True,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        which_agents = self.which_agents
        N_agents = self.N_agents
        
        # extract state-action encoding for each agent
        sa_encodings = []
        for ii in range(N_agents):
            if self.critic_encoders[ii] is not None:
                sa_encodings.append(self.critic_encoders[ii](inps[ii]))
            else:
                sa_encodings.append(None)
            
        # extract state encoding for each agent that we're returning Q for
        s_encodings = []
        for ii in range(N_agents):
            if self.state_encoders[ii] is not None:
                s_encodings.append(self.state_encoders[ii](states[ii]))
            else:
                s_encodings.append(None)
            
        # extract keys for each head for each agent
        all_head_keys = []
        for k_ext in self.key_extractors:
            curr_head = []
            for enc in sa_encodings:
                if enc is not None:
                    curr_head.append(k_ext(enc))
                else:
                    curr_head.append(None)
            all_head_keys.append(curr_head)
                
        #all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]

        # extract sa values for each head for each agent
            
        all_head_values = []
        for v_ext in self.value_extractors:
            curr_head_values = []
            for enc in sa_encodings:
                if enc is not None:
                    curr_head_values.append(v_ext(enc))
                else:
                    curr_head_values.append(None)
            all_head_values.append(curr_head_values)
            
        #all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]

        # extract selectors for each head for each agent that we're returning Q for
            
        all_head_selectors = []
        for sel_ext in self.selector_extractors:
            curr_head_selectors = []
            for enc in s_encodings:
                if enc is not None:
                    curr_head_selectors.append(sel_ext(enc))
                else:
                    curr_head_selectors.append(None)
            all_head_selectors.append(curr_head_selectors)
                
        #all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if which_agents[i]]
        #                      for sel_ext in self.selector_extractors]
            
        other_all_values = []
        all_attend_logits = []
        all_attend_probs = []
        for ii in range(N_agents):
            if which_agents[ii] == True:
                other_all_values.append([])
                all_attend_logits.append([])
                all_attend_probs.append([])
            else:
                other_all_values.append(None)
                all_attend_logits.append(None)
                all_attend_probs.append(None)

            
        #other_all_values = [[] for _ in range(N_agents)]
        #all_attend_logits = [[] for _ in range(N_agents)]
        #all_attend_probs = [[] for _ in range(N_agents)]
        
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            
            # iterate over agents
            for i, a_i, selector in zip(range(N_agents), range(N_agents), curr_head_selectors):

                if selector is not None:

                    keys = [k for j, k in enumerate(curr_head_keys) if ( (j != a_i) and (k is not None) and (j is not None) )]
                    values = [v for j, v in enumerate(curr_head_values) if ( (j != a_i) and (v is not None) and (j is not None) )]

                    # calculate attention across agents
                    attend_logits = torch.matmul(torch.stack(keys), selector.unsqueeze(1))

                    # scale dot-products by size of key (from Attention is All You Need)
                    scaled_attend_logits = attend_logits / np.sqrt(keys[0].detach().numpy().shape[0])
                    attend_weights = F.softmax(scaled_attend_logits, dim=0)
                    other_values = torch.matmul(torch.stack(values).transpose(1,0), attend_weights)

                    other_all_values[a_i].append(other_values)
                    all_attend_logits[a_i].append(attend_logits)
                    all_attend_probs[a_i].append(attend_weights)

        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(range(N_agents)):
            if all_attend_probs[a_i] is not None:

                head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum() ## CHECK...SOMETHING FISHY
                               .mean()) for probs in all_attend_probs[i]]
                agent_rets = []
                NNN = np.prod(np.array(torch.stack(other_all_values[i]).shape))
                critic_in = torch.cat((s_encodings[i], torch.stack(other_all_values[i]).reshape(NNN))) ###CHECK

                q = self.critics[a_i](critic_in)

                if return_q: 
                    agent_rets.append(q)

                if regularize:
                    # regularize magnitude of attention logits
                    attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                    regs = attend_mag_reg
                    agent_rets.append(regs)
                if return_attend:
                    agent_rets.append(np.array(all_attend_probs[i]))
                if logger is not None:
                    logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
                if len(agent_rets) == 1:
                    all_rets.append(agent_rets[0])
                else:
                    all_rets.append(agent_rets)

            else:
                NN = len(all_rets[0])
                if NN == 1:
                    all_rets.append(None)
                else:
                    all_rets.append([None]*NN)

        #pdb.set_trace()    
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets




###### GET INPUT DIMENSIONS STAIGHT FOR CRITICS
class LoanAttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
#    def __init__(self, input_dim_critic1, input_dim_critic2, input_dim_critic3,
#                 input_dim_pol1, input_dim_pol2, input_dim_pol3,
#                 which_agents, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
    def __init__(self, input_dim_critic1, input_dim_critic2, input_dim_critic3,
                 input_dim_pol1, input_dim_pol2, input_dim_pol3,
                 which_agents, fairness_weights, odim= 1, hidden_dim=32, norm_in=True, attend_heads=1):

        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(LoanAttentionCritic, self).__init__()

        assert (hidden_dim % attend_heads) == 0
        #self.sa_sizes = sa_sizes
        #self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics_adm = nn.ModuleList()
        self.critics_disburse = nn.ModuleList()
        self.critics_debt = nn.ModuleList()
        self.critics = []
        self.critic_encoders2 = nn.ModuleList()
        self.critic_loss = []

        # Admission module layers (1 threshold for everyone)
        #self.layer1_admit = nn.Linear(input_dim_critic1, hidden_dim)
        
        if input_dim_critic1 is not None:

            # Critic encoder
            encoder = nn.Sequential()
            encoder.add_module('enc_fc1', nn.Linear(input_dim_critic1, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            encoder.add_module('enc_m1', MEAN())
            self.critic_encoders.append(encoder)

           
            # Critic encoder 2
            critic_encoder2 = nn.Sequential()
            critic_encoder2.add_module('s_enc_fc1', nn.Linear(input_dim_critic1, hidden_dim))
            critic_encoder2.add_module('s_enc_nl', nn.LeakyReLU())
            critic_encoder2.add_module('s_enc_m1', MEAN())
            self.critic_encoders2.append(critic_encoder2)

            # Critic
            #critic = nn.Sequential()
            #critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
            #                                          hidden_dim))
            #critic.add_module('critic_nl', nn.LeakyReLU())
            #critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim)) ##FIX DIMENSIONS LATER!!!!!!!!
            #self.critics.append(critic)
            #self.critic_losses.append(np.array([]))

            #--------Main + Fairness Critics-------
            # Store a dictionary monitoring all critic losses
            curr_critic_losses = {}

            # For main critic
            curr_critics_Main = self.critic_MLP(hidden_dim, odim)
            self.critics_adm.append(curr_critics_Main)
            curr_critic_losses['Main'] = np.array([])
            
            # For Admissions Rates and Score Disparities
            if (fairness_weights[0] is not None) and (fairness_weights[3] is not None):
                
                # Applied_w
                curr_critics_Applied_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Applied_w'] = np.array([])

                # Applied_nw
                curr_critics_Applied_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Applied_nw'] = np.array([])

            else:
                # Applied_w
                curr_critics_Applied_w = None
                curr_critic_losses['Applied_w'] = None
                
                # Applied_nw
                curr_critics_Applied_nw = None
                curr_critic_losses['Applied_nw'] = None
                
            self.critics_adm.append(curr_critics_Applied_w)
            self.critics_adm.append(curr_critics_Applied_nw)

            # For Admissions Rates
            if (fairness_weights[0] is not None):
                
                # Accepted_w
                curr_critics_Accepted_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Accepted_w'] = np.array([])

                # Accepted_nw
                curr_critics_Accepted_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Accepted_nw'] = np.array([])

            else:
                # Accepted_w
                curr_critics_Accepted_w = None
                curr_critic_losses['Accepted_w'] = None
                
                # Accepted_nw
                curr_critics_Accepted_nw = None
                curr_critic_losses['Accepted_nw'] = None
                
            self.critics_adm.append(curr_critics_Accepted_w)
            self.critics_adm.append(curr_critics_Accepted_nw)

            # For Wait Times
            if (fairness_weights[1] is not None):
                
                # WaitTimes_w
                curr_critics_WaitTimes_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['WaitTimes_w'] = np.array([])

                # WaitTimes_nw
                curr_critics_WaitTimes_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['WaitTimes_nw'] = np.array([])

                # Disbursed_w
                curr_critics_Disbursed_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Disbursed_w'] = np.array([])

                # Disbursed_nw
                curr_critics_Disbursed_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Disbursed_nw'] = np.array([])

            else:
                # WaitTimes_w
                curr_critics_WaitTimes_w = None
                curr_critic_losses['WaitTimes_w'] = None
                
                # WaitTimes_nw
                curr_critics_WaitTimes_nw = None
                curr_critic_losses['WaitTimes_nw'] = None

                # Disbursed_w
                curr_critics_Disbursed_w = None
                curr_critic_losses['Disbursed_w'] = None
                
                # Disbursed_nw
                curr_critics_Disbursed_nw = None
                curr_critic_losses['Disbursed_nw'] = None
                
            self.critics_adm.append(curr_critics_WaitTimes_w)
            self.critics_adm.append(curr_critics_WaitTimes_nw)
            self.critics_adm.append(curr_critics_Disbursed_w)
            self.critics_adm.append(curr_critics_Disbursed_nw)

            # For Default Rate
            if (fairness_weights[2] is not None):
                
                # Defaulted_w
                curr_critics_Defaulted_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Defaulted_w'] = np.array([])

                # Defaulted_nw
                curr_critics_Defaulted_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Defaulted_nw'] = np.array([])

                # Terminated_w
                curr_critics_Terminated_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Terminated_w'] = np.array([])

                # Terminated_nw
                curr_critics_Terminated_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Terminated_nw'] = np.array([])

            else:
                # Defaulted_w
                curr_critics_Defaulted_w = None
                curr_critic_losses['Defaulted_w'] = None
                
                # Defaulted_nw
                curr_critics_Defaulted_nw = None
                curr_critic_losses['Defaulted_nw'] = None

                # Terminated_w
                curr_critics_Terminated_w = None
                curr_critic_losses['Terminated_w'] = None
                
                # Terminated_nw
                curr_critics_Terminated_nw = None
                curr_critic_losses['Terminated_nw'] = None
                
            self.critics_adm.append(curr_critics_Defaulted_w)
            self.critics_adm.append(curr_critics_Defaulted_nw)
            self.critics_adm.append(curr_critics_Terminated_w)
            self.critics_adm.append(curr_critics_Terminated_nw)

            # For Score Disparity
            if (fairness_weights[0] is not None):
                
                # Score_Sum_w
                curr_critics_Score_Sum_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Score_Sum_w'] = np.array([])

                # Score_Sum_nw
                curr_critics_Score_Sum_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Score_Sum_nw'] = np.array([])

            else:
                # Score_Sum_w
                curr_critics_Score_Sum_w = None
                curr_critic_losses['Score_Sum_w'] = None
                
                # Score_Sum_nw
                curr_critics_Score_Sum_nw = None
                curr_critic_losses['Score_Sum_nw'] = None
                
            self.critics_adm.append(curr_critics_Score_Sum_w)
            self.critics_adm.append(curr_critics_Score_Sum_nw)

            # Add Admissions critics to list of critics
            self.critics.append(self.critics_adm)
            
            # Store Admissions Critic Losses
            self.critic_loss.append(curr_critic_losses)

                
        else:
            self.critic_encoders.append(None)
            self.critics.append(None)
            #self.state_encoders.append(None)
            self.critic_encoders2.append(None)
            self.critic_loss.append(None)

        if input_dim_critic2 is not None:
            
            # Disbursement module layers (1 threshold for everyone)
            # Critic encoder
            encoder = nn.Sequential()
            encoder.add_module('enc_fc1', nn.Linear(input_dim_critic2, hidden_dim))
            encoder.add_module('enc_n1', nn.LeakyReLU())
            encoder.add_module('enc_m1', MEAN())
            self.critic_encoders.append(encoder)

            # Critic encoder 2
            critic_encoder2 = nn.Sequential()
            critic_encoder2.add_module('s_enc_fc1', nn.Linear(input_dim_critic2, hidden_dim))
            critic_encoder2.add_module('s_enc_nl', nn.LeakyReLU())
            critic_encoder2.add_module('s_enc_m1', MEAN())
            self.critic_encoders2.append(critic_encoder2)

            # Critic
            #critic = nn.Sequential()
            #critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
            #                                          hidden_dim))
            #critic.add_module('critic_nl', nn.LeakyReLU())
            #critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim)) ##FIX DIMENSIONS LATER!!!!!!!!
            #self.critics.append(critic)
            #self.critic_losses.append(np.array([]))

            #--------Main + Fairness Critics-------
            # Store a dictionary monitoring all critic losses
            curr_critic_losses = {}

            # For main critic
            curr_critics_Main = self.critic_MLP(hidden_dim, odim)
            self.critics_disburse.append(curr_critics_Main)
            curr_critic_losses['Main'] = np.array([])
            
            # For Admissions Rates and Score Disparities
            if (fairness_weights[0] is not None) and (fairness_weights[3] is not None):
                
                # Applied_w
                curr_critics_Applied_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Applied_w'] = np.array([])

                # Applied_nw
                curr_critics_Applied_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Applied_nw'] = np.array([])

            else:
                # Applied_w
                curr_critics_Applied_w = None
                curr_critic_losses['Applied_w'] = None
                
                # Applied_nw
                curr_critics_Applied_nw = None
                curr_critic_losses['Applied_nw'] = None
                
            self.critics_disburse.append(curr_critics_Applied_w)
            self.critics_disburse.append(curr_critics_Applied_nw)

            # For Admissions Rates
            if (fairness_weights[0] is not None):
                
                # Accepted_w
                curr_critics_Accepted_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Accepted_w'] = np.array([])

                # Accepted_nw
                curr_critics_Accepted_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Accepted_nw'] = np.array([])

            else:
                # Accepted_w
                curr_critics_Accepted_w = None
                curr_critic_losses['Accepted_w'] = None
                
                # Accepted_nw
                curr_critics_Accepted_nw = None
                curr_critic_losses['Accepted_nw'] = None
                
            self.critics_disburse.append(curr_critics_Accepted_w)
            self.critics_disburse.append(curr_critics_Accepted_nw)

            # For Wait Times
            if (fairness_weights[1] is not None):
                
                # WaitTimes_w
                curr_critics_WaitTimes_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['WaitTimes_w'] = np.array([])

                # WaitTimes_nw
                curr_critics_WaitTimes_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['WaitTimes_nw'] = np.array([])

                # Disbursed_w
                curr_critics_Disbursed_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Disbursed_w'] = np.array([])

                # Disbursed_nw
                curr_critics_Disbursed_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Disbursed_nw'] = np.array([])

            else:
                # WaitTimes_w
                curr_critics_WaitTimes_w = None
                curr_critic_losses['WaitTimes_w'] = None
                
                # WaitTimes_nw
                curr_critics_WaitTimes_nw = None
                curr_critic_losses['WaitTimes_nw'] = None

                # Disbursed_w
                curr_critics_Disbursed_w = None
                curr_critic_losses['Disbursed_w'] = None
                
                # Disbursed_nw
                curr_critics_Disbursed_nw = None
                curr_critic_losses['Disbursed_nw'] = None
                
            self.critics_disburse.append(curr_critics_WaitTimes_w)
            self.critics_disburse.append(curr_critics_WaitTimes_nw)
            self.critics_disburse.append(curr_critics_Disbursed_w)
            self.critics_disburse.append(curr_critics_Disbursed_nw)


            # For Default Rate
            if (fairness_weights[2] is not None):
                
                # Defaulted_w
                curr_critics_Defaulted_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Defaulted_w'] = np.array([])

                # Defaulted_nw
                curr_critics_Defaulted_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Defaulted_nw'] = np.array([])

                # Terminated_w
                curr_critics_Terminated_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Terminated_w'] = np.array([])

                # Terminated_nw
                curr_critics_Terminated_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Terminated_nw'] = np.array([])

            else:
                # Defaulted_w
                curr_critics_Defaulted_w = None
                curr_critic_losses['Defaulted_w'] = None
                
                # Defaulted_nw
                curr_critics_Defaulted_nw = None
                curr_critic_losses['Defaulted_nw'] = None

                # Terminated_w
                curr_critics_Terminated_w = None
                curr_critic_losses['Terminated_w'] = None
                
                # Terminated_nw
                curr_critics_Terminated_nw = None
                curr_critic_losses['Terminated_nw'] = None
                
            self.critics_disburse.append(curr_critics_Defaulted_w)
            self.critics_disburse.append(curr_critics_Defaulted_nw)
            self.critics_disburse.append(curr_critics_Terminated_w)
            self.critics_disburse.append(curr_critics_Terminated_nw)

            # For Score Disparity
            if (fairness_weights[0] is not None):
                
                # Score_Sum_w
                curr_critics_Score_Sum_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Score_Sum_w'] = np.array([])

                # Score_Sum_nw
                curr_critics_Score_Sum_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Score_Sum_nw'] = np.array([])

            else:
                # Score_Sum_w
                curr_critics_Score_Sum_w = None
                curr_critic_losses['Score_Sum_w'] = None
                
                # Score_Sum_nw
                curr_critics_Score_Sum_nw = None
                curr_critic_losses['Score_Sum_nw'] = None
                
            self.critics_disburse.append(curr_critics_Score_Sum_w)
            self.critics_disburse.append(curr_critics_Score_Sum_nw)

            # Add Disbursement critics to list of critics
            self.critics.append(self.critics_disburse)

            # Store Disbursement Critic Losses
            self.critic_loss.append(curr_critic_losses)
                            
        else:
            self.critic_encoders.append(None)
            self.critics.append(None)
            self.critic_encoders2.append(None)
            self.critic_loss.append(None)

        if input_dim_critic3 is not None:
            
            # Debt module layers (1 threshold for everyone)
            # Critic encoder
            encoder = nn.Sequential()
            encoder.add_module('enc_fc1', nn.Linear(input_dim_critic3, hidden_dim))
            encoder.add_module('enc_n1', nn.LeakyReLU())
            encoder.add_module('enc_m1', MEAN())
            self.critic_encoders.append(encoder)

            # Critic encoder 2
            critic_encoder2 = nn.Sequential()
            critic_encoder2.add_module('s_enc_fc1', nn.Linear(input_dim_critic3, hidden_dim))
            critic_encoder2.add_module('s_enc_nl', nn.LeakyReLU())
            critic_encoder2.add_module('s_enc_m1', MEAN())
            self.critic_encoders2.append(critic_encoder2)

            # Critic
            #critic = nn.Sequential()
            #critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
            #                                          hidden_dim))
            #critic.add_module('critic_nl', nn.LeakyReLU())
            #critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim)) ##FIX DIMENSIONS LATER!!!!!!!!
            #self.critics.append(critic)
            #self.critic_loss.append(np.array([]))

            #--------Main + Fairness Critics-------
            # Store a dictionary monitoring all critic losses
            curr_critic_losses = {}

            # For main critic
            curr_critics_Main = self.critic_MLP(hidden_dim, odim)
            self.critics_debt.append(curr_critics_Main)
            curr_critic_losses['Main'] = np.array([])
            
            # For Admissions Rates and Score Disparities
            if (fairness_weights[0] is not None) and (fairness_weights[3] is not None):
                
                # Applied_w
                curr_critics_Applied_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Applied_w'] = np.array([])

                # Applied_nw
                curr_critics_Applied_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Applied_nw'] = np.array([])

            else:
                # Applied_w
                curr_critics_Applied_w = None
                curr_critic_losses['Applied_w'] = None
                
                # Applied_nw
                curr_critics_Applied_nw = None
                curr_critic_losses['Applied_nw'] = None
                
            self.critics_debt.append(curr_critics_Applied_w)
            self.critics_debt.append(curr_critics_Applied_nw)

            # For Admissions Rates
            if (fairness_weights[0] is not None):
                
                # Accepted_w
                curr_critics_Accepted_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Accepted_w'] = np.array([])

                # Accepted_nw
                curr_critics_Accepted_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Accepted_nw'] = np.array([])

            else:
                # Accepted_w
                curr_critics_Accepted_w = None
                curr_critic_losses['Accepted_w'] = None
                
                # Accepted_nw
                curr_critics_Accepted_nw = None
                curr_critic_losses['Accepted_nw'] = None
                
            self.critics_debt.append(curr_critics_Accepted_w)
            self.critics_debt.append(curr_critics_Accepted_nw)

            # For Wait Times
            if (fairness_weights[1] is not None):
                
                # WaitTimes_w
                curr_critics_WaitTimes_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['WaitTimes_w'] = np.array([])

                # WaitTimes_nw
                curr_critics_WaitTimes_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['WaitTimes_nw'] = np.array([])

                # Disbursed_w
                curr_critics_Disbursed_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Disbursed_w'] = np.array([])

                # Disbursed_nw
                curr_critics_Disbursed_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Disbursed_nw'] = np.array([])

            else:
                # WaitTimes_w
                curr_critics_WaitTimes_w = None
                curr_critic_losses['WaitTimes_w'] = None
                
                # WaitTimes_nw
                curr_critics_WaitTimes_nw = None
                curr_critic_losses['WaitTimes_nw'] = None

                # Disbursed_w
                curr_critics_Disbursed_w = None
                curr_critic_losses['Disbursed_w'] = None
                
                # Disbursed_nw
                curr_critics_Disbursed_nw = None
                curr_critic_losses['Disbursed_nw'] = None
                
            self.critics_debt.append(curr_critics_WaitTimes_w)
            self.critics_debt.append(curr_critics_WaitTimes_nw)
            self.critics_debt.append(curr_critics_Disbursed_w)
            self.critics_debt.append(curr_critics_Disbursed_nw)


            # For Default Rate
            if (fairness_weights[2] is not None):
                
                # Defaulted_w
                curr_critics_Defaulted_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Defaulted_w'] = np.array([])

                # Defaulted_nw
                curr_critics_Defaulted_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Defaulted_nw'] = np.array([])

                # Terminated_w
                curr_critics_Terminated_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Terminated_w'] = np.array([])

                # Terminated_nw
                curr_critics_Terminated_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Terminated_nw'] = np.array([])

            else:
                # Defaulted_w
                curr_critics_Defaulted_w = None
                curr_critic_losses['Defaulted_w'] = None
                
                # Defaulted_nw
                curr_critics_Defaulted_nw = None
                curr_critic_losses['Defaulted_nw'] = None

                # Terminated_w
                curr_critics_Terminated_w = None
                curr_critic_losses['Terminated_w'] = None
                
                # Terminated_nw
                curr_critics_Terminated_nw = None
                curr_critic_losses['Terminated_nw'] = None
                
            self.critics_debt.append(curr_critics_Defaulted_w)
            self.critics_debt.append(curr_critics_Defaulted_nw)
            self.critics_debt.append(curr_critics_Terminated_w)
            self.critics_debt.append(curr_critics_Terminated_nw)

            # For Score Disparity
            if (fairness_weights[0] is not None):
                
                # Score_Sum_w
                curr_critics_Score_Sum_w = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Score_Sum_w'] = np.array([])

                # Score_Sum_nw
                curr_critics_Score_Sum_nw = self.critic_MLP(hidden_dim, odim)
                curr_critic_losses['Score_Sum_nw'] = np.array([])

            else:
                # Score_Sum_w
                curr_critics_Score_Sum_w = None
                curr_critic_losses['Score_Sum_w'] = None
                
                # Score_Sum_nw
                curr_critics_Score_Sum_nw = None
                curr_critic_losses['Score_Sum_nw'] = None
                
            self.critics_debt.append(curr_critics_Score_Sum_w)
            self.critics_debt.append(curr_critics_Score_Sum_nw)

            # Add Debt critic to list of critics
            self.critics.append(self.critics_debt)

            # Store Debt Critic Losses
            self.critic_loss.append(curr_critic_losses)
            
        else:
            self.critic_encoders.append(None)
            self.critics.append(None)
            self.critic_encoders2.append(None)
            self.critic_loss.append(None)

        # Merge module layers
        self.fairness_weights = fairness_weights
        self.which_agents = which_agents
        self.N_agents = 3

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

        ### EVERYTHING UP TO THIS POINT IN THE INIT SHOULD BE CONCEPTUALLY OKAY. STILL NEED TO TEST THOUGH, BUT START CODING NEXT PART FIRST####

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def critic_MLP(self, hidden_dim, odim):
        critic = nn.Sequential()
        critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
        critic.add_module('critic_nl', nn.LeakyReLU())
        critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim)) ##FIX DIMENSIONS LATER!!!!!!!!

        return critic

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.N_agents)

    def forward(self, states, actions, inps, agents=None, return_q=True,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        which_agents = self.which_agents
        N_agents = self.N_agents
        
        # extract state-action encoding for each agent
        sa_encodings = []
        for ii in range(N_agents):
            if self.critic_encoders[ii] is not None:
                sa_encodings.append(self.critic_encoders[ii](inps[ii]))
            else:
                sa_encodings.append(None)
            
        # extract state encoding for each agent that we're returning Q for
        sa2_encodings = []
        for ii in range(N_agents):
            if self.critic_encoders2[ii] is not None:
                #s_encodings.append(self.state_encoders[ii](states[ii]))
                sa2_encodings.append(self.critic_encoders2[ii](inps[ii]))
            else:
                #s_encodings.append(None)
                sa2_encodings.append(None)
            
        # extract keys for each head for each agent
        all_head_keys = []
        for k_ext in self.key_extractors:
            curr_head = []
            for enc in sa_encodings:
                if enc is not None:
                    curr_head.append(k_ext(enc))
                else:
                    curr_head.append(None)
            all_head_keys.append(curr_head)
                
        #all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]

        # extract sa values for each head for each agent
            
        all_head_values = []
        for v_ext in self.value_extractors:
            curr_head_values = []
            for enc in sa_encodings:
                if enc is not None:
                    curr_head_values.append(v_ext(enc))
                else:
                    curr_head_values.append(None)
            all_head_values.append(curr_head_values)
            
        #all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]

        # extract selectors for each head for each agent that we're returning Q for
            
        all_head_selectors = []
        for sel_ext in self.selector_extractors:
            curr_head_selectors = []
            for enc in sa2_encodings:
                if enc is not None:
                    #curr_head_selectors.append(sel_ext(enc))
                    curr_head_selectors.append(sel_ext(enc))
                else:
                    curr_head_selectors.append(None)
            all_head_selectors.append(curr_head_selectors)
                
        #all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if which_agents[i]]
        #                      for sel_ext in self.selector_extractors]
            
        other_all_values = []
        all_attend_logits = []
        all_attend_probs = []
        for ii in range(N_agents):
            if which_agents[ii] == True:
                other_all_values.append([])
                all_attend_logits.append([])
                all_attend_probs.append([])
            else:
                other_all_values.append(None)
                all_attend_logits.append(None)
                all_attend_probs.append(None)

            
        #other_all_values = [[] for _ in range(N_agents)]
        #all_attend_logits = [[] for _ in range(N_agents)]
        #all_attend_probs = [[] for _ in range(N_agents)]
        
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            
            # iterate over agents
            for i, a_i, selector in zip(range(N_agents), range(N_agents), curr_head_selectors):

                if selector is not None:

                    keys = [k for j, k in enumerate(curr_head_keys) if ( (j != a_i) and (k is not None) and (j is not None) )]        # Other agents are the keys
                    #values = [v for j, v in enumerate(curr_head_values) if ( (j != a_i) and (v is not None) and (j is not None) )]
                    values = [v for j, v in enumerate(curr_head_values) if ( (j != a_i) and (v is not None) and (j is not None) )]     # Current agent is the query

                    # calculate attention across agents
                    attend_logits = torch.matmul(torch.stack(keys), selector.unsqueeze(1))

                    # scale dot-products by size of key (from Attention is All You Need)
                    scaled_attend_logits = attend_logits / np.sqrt(keys[0].detach().numpy().shape[0])
                    attend_weights = F.softmax(scaled_attend_logits, dim=0)
                    other_values = torch.matmul(torch.stack(values).transpose(1,0), attend_weights)

                    other_all_values[a_i].append(other_values)
                    all_attend_logits[a_i].append(attend_logits)
                    all_attend_probs[a_i].append(attend_weights)

        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(range(N_agents)):
            if all_attend_probs[a_i] is not None:

                head_entropies = [(-((probs + 1e-8).log() * probs).squeeze().sum() ## CHECK...SOMETHING FISHY
                               .mean()) for probs in all_attend_probs[i]]
                agent_rets = []
                NNN = np.prod(np.array(torch.stack(other_all_values[i]).shape))
                critic_in = torch.cat((sa_encodings[i], torch.stack(other_all_values[i]).reshape(NNN))) ###CHECK

                curr_agent_rets = {}

                # Main critic
                q = self.critics[a_i][0](critic_in)
                curr_agent_rets['Main'] = q

                # For Admissions Rates and Score Disparities
                if (self.fairness_weights[0] is not None) or (self.fairness_weights[3] is not None):

                    q = self.critics[a_i][1](critic_in)
                    curr_agent_rets['Applied_w'] = q

                    q = self.critics[a_i][2](critic_in)
                    curr_agent_rets['Applied_nw'] = q

                else:
                    curr_agent_rets[3] = None
                    curr_agent_rets['Applied_nw'] = None

                # For Admissions Rates
                if (self.fairness_weights[0] is not None):

                    q = self.critics[a_i][3](critic_in)
                    curr_agent_rets['Accepted_w'] = q

                    q = self.critics[a_i][4](critic_in)
                    curr_agent_rets['Accepted_nw'] = q

                else:
                    curr_agent_rets['Accepted_w'] = None
                    curr_agent_rets['Accepted_nw'] = None


                # For WaitTimes
                if (self.fairness_weights[1] is not None):

                    q = self.critics[a_i][5](critic_in)
                    curr_agent_rets['WaitTimes_w'] = q

                    q = self.critics[a_i][6](critic_in)
                    curr_agent_rets['WaitTimes_nw'] = q

                    q = self.critics[a_i][7](critic_in)
                    curr_agent_rets['Disbursed_w'] = q

                    q = self.critics[a_i][8](critic_in)
                    curr_agent_rets['Disbursed_nw'] = q

                else:
                    curr_agent_rets['WaitTimes_w'] = None
                    curr_agent_rets['WaitTimes_nw'] = None
                    curr_agent_rets['Disbursed_w'] = None
                    curr_agent_rets['Disbursed_nw'] = None


                # For Default Rate
                if (self.fairness_weights[2] is not None):

                    q = self.critics[a_i][9](critic_in)
                    curr_agent_rets['Defaulted_w'] = q

                    q = self.critics[a_i][10](critic_in)
                    curr_agent_rets['Defaulted_nw'] = q

                    q = self.critics[a_i][11](critic_in)
                    curr_agent_rets['Terminated_w'] = q

                    q = self.critics[a_i][12](critic_in)
                    curr_agent_rets['Terminated_nw'] = q

                else:
                    curr_agent_rets['Defaulted_w'] = None
                    curr_agent_rets['Defaulted_nw'] = None
                    curr_agent_rets['Terminated_w'] = None
                    curr_agent_rets['Terminated_nw'] = None

                # For Score Disparity
                if (self.fairness_weights[3] is not None):

                    q = self.critics[a_i][13](critic_in)
                    curr_agent_rets['Score_Sum_w'] = q

                    q = self.critics[a_i][14](critic_in)
                    curr_agent_rets['Score_Sum_nw'] = q

                else:
                    curr_agent_rets['Score_Sum_w'] = None
                    curr_agent_rets['Score_Sum_nw'] = None

                if return_q: 
                    agent_rets.append(curr_agent_rets)

                if regularize:
                    # regularize magnitude of attention logits
                    attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                    regs = attend_mag_reg
                    agent_rets.append(regs)
                if return_attend:
                    agent_rets.append(np.array(all_attend_probs[i]))
                if logger is not None:
                    logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
                if len(agent_rets) == 1:
                    all_rets.append(agent_rets[0])
                else:
                    all_rets.append(agent_rets)

            else:
                NN = len(all_rets[0])
                if NN == 1:
                    all_rets.append(None)
                else:
                    all_rets.append([None]*NN)

        #pdb.set_trace()    
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets





        










# Wrapping a mean layer for sequential
class MEAN(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=0)
