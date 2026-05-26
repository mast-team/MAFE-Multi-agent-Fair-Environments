import scenarios
import pdb


def make_env_04082024(scenario_name, policyTypes, which_agents, AwareOrNot, eq_flags, adm_freq, aug_state_flag=False):

    # load scenario script
    env = scenarios.load(scenario_name + ".py").Environment(policyTypes, which_agents, AwareOrNot, eq_flags, adm_freq, aug_state_flag=aug_state_flag)

    return env
