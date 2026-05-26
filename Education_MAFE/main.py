import argparse
import torch
import os
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import pdb
from algorithms.multiagent_algos import CrossEntropyMethod
from make_env import make_env
import glob
import _pickle as pickle
import gzip
import warnings

def get_agent_scenario(env_id, policyTypes):

    # Which agents are RL models (True) and which agents
    # part of the environment (False)

    which_agents = []

    # University Admissions Agent
    if policyTypes[0] == None:
        which_agents.append(False)
    else:
        which_agents.append(True)

    # University Budget Allocation Agent
    if policyTypes[1] == None:
        which_agents.append(False)
    else:
        which_agents.append(True)

    # Employer agent
    if policyTypes[2] == None:
        which_agents.append(False)
    else:
        which_agents.append(True)

    # Planner agent
    if policyTypes[3] == None:
        which_agents.append(False)
    else:
        which_agents.append(True)

    return which_agents

# Code that handles model specific update goes here
def model_specific_update(env, model, obs, which_agents, other_params, replay_buffer=None):

    episode_length = other_params['episode_length'] # How many time steps in episode
    n_episodes = other_params['n_episodes']
    N_episodes = other_params['N_episodes'] # How many episodes in an epoch
    epoch = other_params['epoch']  # current epoch
    N_episodes = other_params['N_episodes']
    N_epochs = other_params['N_epochs']   # How many total training epochs
    tt = other_params['tt']
    store_rewards_dir = other_params['store_rewards_dir']
    trial_num = other_params['trial_num']

    # Start of an episode
    if tt == 0:

        print("Starting Episode %i of Epoch %i" % (n_episodes, epoch))
        
        # Reset World
        obs = env.reset_world()

        # Sample model parameters
        model.sample_intervention_parameters()

        # Update time step in episode
        tt += 1

        # Update parameters
        other_params['tt'] = tt
        return obs, env, model, other_params

    # Get actions
    actions = model.GetActions(obs)

    # Import actions to environment
    # If there is an university admissions agent, update its state
    if which_agents[0]:
        env.world.agents[0].action = actions[0]
        
    # If there is a university budget allocation agent, update its state
    if which_agents[1]:
        env.world.agents[1].action = actions[1]

        # Save actions
        if actions[1] is not None and len(actions[1].flatten()) > 0:
            model.ActionTrackers["univ_budget_actions"][epoch, n_episodes, tt-1, :] = actions[1].flatten()
        else:
            N_sz = len(model.ActionTrackers["univ_budget_actions"][epoch, n_episodes, tt-1, :])
            model.ActionTrackers["univ_budget_actions"][epoch, n_episodes, tt-1, :] = np.ones(( N_sz ))*-1

    # If there is an employer agent, update its state
    if which_agents[2]:
        env.world.agents[2].action = actions[2]

        # Save actions
        if actions[2] is not None and len(actions[2].flatten()) > 0:
            model.ActionTrackers["emp_actions"][epoch, n_episodes, tt-1, :] = np.array([np.mean(actions[2]), np.var(actions[2])])
        else:
            model.ActionTrackers["emp_actions"][epoch, n_episodes, tt-1, :] = np.ones((1))*-1


    # If there is a planner agent, update its state
    if which_agents[3]:
        env.world.agents[3].action = actions[3]
        
        # Save actions
        if actions[3] is not None and len(actions[3].flatten()) > 0:
            model.ActionTrackers["planner_actions"][epoch, n_episodes, tt-1, :] = actions[3].flatten()
        else:
            N_sz = len(model.ActionTrackers["planner_actions"][epoch, n_episodes, tt-1, :])
            model.ActionTrackers["planner_actions"][epoch, n_episodes, tt-1, :] = np.ones(( N_sz ))*-1


    # Update environment
    obs, reward, failed_ep, _ = env.step(tt-1)

    # Stack rewards
    model.reward_t = np.vstack((model.reward_t, reward))
    
    # Update time step in episode
    tt += 1

    # Check for failure (If failed, bad episode to training...restart)
    if failed_ep:
        model.reset_episode()
        tt = 0
        
    # End of an episode
    if tt == episode_length:

        # Store rewards from this episode
        model.update_rewards()

        # Store sampled weights from current iteration for elite set calculation
        model.DictRealizations_N.append(model.AgentRealization)

        # Reset model for next episode
        model.reset_episode()
        n_episodes += 1
        
        tt = 0 # Envirnoment reset

    # End of epoch
    if n_episodes == N_episodes: 
        
        # Update Epoch
        model.step()
        model.reset_epoch()
        epoch += 1
        model.epoch = epoch
        n_episodes = 0

        # Model storage
        store_model_file = store_rewards_dir + "trial_"+ str(trial_num) + "/model.pkl"
        file_model = open(store_model_file, 'wb')
        pickle.dump(model, file_model)
    

    # Update parameters
    other_params['epoch']=epoch
    other_params['n_episodes'] = n_episodes
    other_params['tt'] = tt
    if epoch == N_epochs:
        other_params['termination_flag'] = True

    return obs, env, model, other_params


def run(config):

    if config.SEED is not None:
        run_num = config.SEED
    else:
        run_num = 0
        
    which_agents = get_agent_scenario(config.env_id, config.policyTypes)
    torch.manual_seed(run_num)
    np.random.seed(run_num)    
    env = make_env(config.env_id,
                   config.policyTypes,
                   which_agents,
                   config.k_cycle,
                   config.episode_length)

    env.reset_world()

    # Initialize parameter storage
    other_params = {}

    # If new trial create new directory for it
    store_rewards_dir = os.getcwd() + "/stored_rewards_data/Profits_fairness/CrossEntropyMethod/"
    if config.trial_num is None:
        trial_num = len(glob.glob1(store_rewards_dir,"trial_*"))
        os.mkdir(store_rewards_dir + "trial_"+ str(trial_num) + "/")

        # Convert inputs to model specs
        model = CrossEntropyMethod(env              = env,
                                    policyTypes      = config.policyTypes,
                                    lmds             = config.lmd_all,
                                    tracking_path    = store_rewards_dir + "trial_"+ str(trial_num) + "/",
                                    N_epochs         = config.N_epochs,
                                    episode_length   = config.episode_length,
                                    opt_type         = config.opt_type,
                                    which_agents     = which_agents,
                                    N_episodes       = config.N_episodes)
        other_params['epoch'] = 0

        # Store meta data associated with this training session
        file_meta_dta = store_rewards_dir + "trial_"+ str(trial_num) + "/meta_dta.txt"
        with open(file_meta_dta,'w') as out:
            line0 = "Training Method: "+str(config.model_type)
            line1 = "Weights for each element of loss (e.g. lmds): "+str(config.lmd_all)
            line2 = "Episode length: "+str(config.episode_length)
            line3 = "Training method subtype: "+config.opt_type
            line4 = "Number of epochs: "+str(config.N_epochs)
            line5 = "Number of episodes per training update: "+str(config.N_episodes)
            line6 = "Policy Types: "+str(config.policyTypes)
            line7 = "Cycles for each agent actions: " + str(config.k_cycle)
            line8 = "Seed: " + str(run_num)
            out.write('{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n'.format(line0,line1,line2,line3,line4,line5,line6,line7,line8))
        
    else:
        trial_num = config.trial_num
        file_name_model = store_rewards_dir + "trial_"+ str(trial_num) + "/model.pkl"
        file_model = open(file_name_model, 'rb')
        model = pickle.load(file_model)
        
        # Make sure agents selected by user match the ones saved by the model
        if config.policyTypes != model.policyTypes:
            ValueError("policyTypes incompatible with saved model")

        other_params['epoch'] = model.epoch

    # Reset model
    model.reset_epoch()
    model.reset_episode()

    # Store parameters
    other_params['termination_flag'] = False
    other_params['episode_length'] = config.episode_length
    other_params['N_episodes'] = config.N_episodes
    other_params['N_epochs'] = config.N_epochs
    other_params['n_episodes'] = 0
    other_params['tt'] = 0
    other_params['store_rewards_dir'] = store_rewards_dir
    other_params['trial_num'] = trial_num
    obs = None
    
    while not other_params['termination_flag']:

        obs, env, model, other_params = model_specific_update(env, model, obs, which_agents, other_params)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("--model_type",
                        help="Name of directory to store " +
                             "model/training contents", default="CrossEntropyMethod", type=str)

    parser.add_argument("--hidden_dim", default=8, type=int)
    parser.add_argument("--pol_hidden_dim", default=8, type=int)
    parser.add_argument("--critic_hidden_dim", default=8, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.01, type=float)
    parser.add_argument("--q_lr", default=0.01, type=float)
    parser.add_argument("--lr_critic", default=0.01, type=float)
    parser.add_argument("--lr_policy", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--agent_scenario_id", default=3, type=int)

    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--agent_names", default="univ_adm,univ_budget,emp,planner",
                            help="Names of the different interventions included in rewards")
    parser.add_argument("--intervention_names", default="emp_profits,admRts,gradRts_UG,gradRts_MS,gradRts_PH,admRts_minorEqty,gradRts_UG_minorEqty,gradRts_MS_minorEqty,gradRts_PH_minorEqty,salary_minorEqty",
                            help="Names of the different interventions included in rewards")
    parser.add_argument("--lmds", default="0.11111,0.11111,0.11111,0.11111,0.11111,0.11111,0.11111,0.11111,0.11111,0.11111", type=str)
    parser.add_argument("--N_epochs", default=50, type=int)
    parser.add_argument("--N_episodes", default=100, type=int)
    parser.add_argument("--k_cyclic", default="1,1,1,1", type=str,
                            help="0: cycles given by k_cycle[0], 1: cycles given by k_cycle[1]")
    parser.add_argument("--trial_num", default="None", type=str)
    parser.add_argument("--policyTypes", default="1,1,1,1", type=str)
    parser.add_argument("--opt_type", default="cooperate", type=str)

    parser.add_argument("--N_ep_per_batch", default=100, type=int,
                            help="Number of episodes in rollout buffer")

    parser.add_argument("--Fair_scheme_AC", default=0, type=int,
                            help="0: Furong's method, 1: accumulated direct rates")
    parser.add_argument("--SEED", default="None", type=str)

    config = parser.parse_args()

    # Update list items
    config.policyTypes          = config.policyTypes.split(',')
    config.agent_names          = config.agent_names.split(',')
    config.intervention_names   = config.intervention_names.split(',')
    config.lmds                 = config.lmds.split(',')
    config.k_cyclic             = config.k_cyclic.split(',')
    config.k_cycle              = {}
    config.lmd_all              = {}

    # Interpret commandline inputs
    if config.trial_num == "None":
        config.trial_num = None
    else:
        config.trial_num = int(config.trial_num)

    if config.SEED == "None":
        config.SEED = None
    else:
        config.SEED = int(config.SEED)

    # Gather lmds
    for ii in range(len(config.lmds)):
        if config.lmds[ii] == 'None':
            config.lmd_all[config.intervention_names[ii]] = None
        else:
            config.lmd_all[config.intervention_names[ii]] = float(config.lmds[ii])

    # Gather cycles
    for ii in range(len(config.k_cyclic)):
        if config.k_cyclic[ii] == 'None':
            config.k_cycle[config.agent_names[ii]] = None
        else:
            config.k_cycle[config.agent_names[ii]] = int(config.k_cyclic[ii])

    # Policy types
    for ii in range(len(config.policyTypes)):
        if config.policyTypes[ii] == 'None':
            config.policyTypes[ii] = None
        else:
            config.policyTypes[ii] = int(config.policyTypes[ii])

    warnings.filterwarnings('ignore')

    run(config)















    
