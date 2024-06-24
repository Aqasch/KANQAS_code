import configparser
import json
from warnings import simplefilter 
simplefilter(action='ignore', category=DeprecationWarning)
from itertools import product

def get_config(config_name,experiment_name, path='configuration_files',
               verbose=True):
    config_dict = {}
    Config = configparser.ConfigParser()
    Config.read('{}/{}{}'.format(path,config_name,experiment_name))
    for sections in Config:
        config_dict[sections] = {}
        for key, val in Config.items(sections):
            # config_dict[sections].update({key: json.loads(val)})
            try:
                config_dict[sections].update({key: int(val)})
            except ValueError:
                config_dict[sections].update({key: val})
            floats = ['learning_rate',  'dropout', 'alpha', 
                      'beta', 'beta_incr', 
                      "shift_threshold_ball","succes_switch","tolearance_to_thresh","memory_reset_threshold",
                      "fake_min_energy","_true_en"]
            strings = ['ham_type', 'fn_type', 'geometry','method','agent_type',
                       "agent_class","init_seed","init_path","init_thresh","method",
                       "mapping","optim_alg", "curriculum_type"]
            lists = ['episodes','neurons', 'accept_err','epsilon_decay',"epsilon_min",
                     "epsilon_decay",'final_gamma','memory_clean',
                     'update_target_net', 'epsilon_restart', "thresholds", "switch_episodes"]
            if key in floats:
                config_dict[sections].update({key: float(val)})
            elif key in strings:
                config_dict[sections].update({key: str(val)})
            elif key in lists:
                config_dict[sections].update({key: json.loads(val)})
    del config_dict['DEFAULT']
    return config_dict

def dictionary_of_actions(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations.
    """
    dictionary = dict()
    i = 0
         
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
   
    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits),
                           range(1, 6)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary
        
def dict_of_actions_revert_q(num_qubits):
    """
    Creates dictionary of actions for system which steers positions of gates,
    and axes of rotations. Systems have reverted order to above dictionary of actions.
    """
    dictionary = dict()
    i = 0
         
    for c, x in product(range(num_qubits-1,-1,-1),
                        range(num_qubits-1,0,-1)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
   
    """h  denotes rotation axis. 1, 2, 3 -->  X, Y, Z axes """
    for r, h in product(range(num_qubits-1,-1,-1),
                           range(1, 6)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary
