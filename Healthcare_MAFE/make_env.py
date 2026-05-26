import scenarios

def make_env(scenario_name, policyTypes, which_agents, k_cycle, episode_length):

    # load scenario script
    env = scenarios.load(scenario_name + ".py").Environment(policyTypes, which_agents, k_cycle, episode_length)

    return env
