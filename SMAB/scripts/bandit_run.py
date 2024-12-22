#python bandit_run.py -s "oldarms_predictedprob_visit1_30000_Thompson/" -r "/home/debrupd1/XCOPA/hate_test_predictions/error_fixed_arms_MLM_mBERT_withprobs.json" -m 'yes' -algo 'thompson' -prob 'yes'
import os 
import re
import time
import json
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from bandits import Bandit
from collections import defaultdict, OrderedDict
import argparse

def write_file(dict_name, file_name):
    with open(file_name, "a") as fl:
        json.dump(dict_name, fl)
        fl.write("\n")

def ensure_directory_exists(path):
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory {path} created successfully or already exists.")
    except Exception as e:
        print(f"An error occurred while creating the directory {path}: {e}")


def get_tup_hash(data_reward):

    hash_ = defaultdict(list)
    highest_reward = defaultdict(list)
    tup = ()

    for i in range(len(data_reward)):

        row_temp = data_reward[i]['meta']
        lowercase_word = row_temp['word'].lower()
        if lowercase_word not in tup:
            tup = tup + (lowercase_word,)

        hash_[lowercase_word].extend(row_temp['samples'])
        
    return tup, hash_


'''
for k,_ in hash.items():
    h_reward = 0
    samples = hash[k][0]
    print(samples)
    print(len(samples))
    
    for j in range(len(samples)):
        if h_reward < samples[j][2]:
            h_reward = samples[j][2]
    highest_reward[k].append(h_reward)

'''


def test(t, d):

    assert len(t) == len(d)
    count = 0
    for k, _ in d.items():
        assert k == t[count]
        count += 1
    print("test finished")




def write_global_sensitivity_initvalues(tup, bandit_setup, working_dir, setting):
    dict_initial_sensitivity = {}
    for k in range(len(tup)):
        dict_initial_sensitivity[tup[k]] = bandit_setup.global_sensitivity[k]
    assert len(dict_initial_sensitivity) == len(tup)
    write_file(dict_initial_sensitivity, working_dir + setting + "global_sensitivity_initialvalues.json")
    print("finished writing initial global sensitivity")


def write_global_sensitivity_finalvalues(tup, bandit_setup, working_dir, setting):
    dict_final_sensitivity = {}
    for k in range(len(tup)):
        #print(bandit_setup.global_sensitivity[k])
        dict_final_sensitivity[tup[k]] = bandit_setup.global_sensitivity[k]
    assert len(dict_final_sensitivity) == len(tup)
    write_file(dict_final_sensitivity, working_dir + setting + "global_sensitivity_finalvalues.json")
    print("finished writing final global sensitivity")

def write_regret_per_iteration(tup, bandit_setup, working_dir, setting):
    dict_regret = {}
    for k in range(len(bandit_setup.regret_list)):
        dict_regret[k] = bandit_setup.regret_list[k]
    write_file(dict_regret, working_dir + setting + "regret_per_iteration.json")

def write_regret_freq_localsens(tup, bandit_setup, working_dir, setting):
    '''
    dict_regret = {}
    for k in range(len(bandit_setup.regret_list)):
        dict_regret[k] = bandit_setup.regret_list[k]
    write_file(dict_regret, working_dir + "regret_per_iteration.json")
    print("finished writing regret_per_iteration")
    '''

    dict_rc_tuple = dict(bandit_setup.reg_count_tuple)
    write_file(dict_rc_tuple, working_dir + setting + "regret_count_tuple.json")
    print("finished writing regret_count_tuple")

    dict_count = {}
    for k in range(len(tup)):
        dict_count[tup[k]] = bandit_setup.count[k]

    write_file(dict_count, working_dir + setting + "word_frequency.json")
    print("finished writing word_frequency")

    print("No of single samples:", bandit_setup.count_singlesample)

    # Store the list of dictionaries
    d_localsens = bandit_setup.local_sens
    with open(working_dir + setting + 'localsens_of_dicts.json', 'w') as file:
        json.dump(d_localsens, file)


    
if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-s", "--setting", help = "Setting", default="allcombinations/")
    parser.add_argument("-algo", "--algorithm", help = "MAB Algorithm", default="ucb")
    parser.add_argument("-i", "--num_iters", type=int, help = "Number of iterations", default=50000)
    parser.add_argument("-r", "--reward_file", help = "Reward file", default="data/eng/new/xlmr_reward_withoutgold_mbert_eng.json")
    parser.add_argument("-d", "--data_file", help = "Data file", default="data/eng/new/xlmr_reward_withoutgold_mbert_eng.json")
    parser.add_argument("-l", "--llama", help = "Use llama for generation", default="no")
    parser.add_argument("-m", "--mbert_mlm", help = "Using mBert MLM", default="no")
    parser.add_argument("-prob", "--probreward", help = "Whether to use prob for reward", default="no")
    parser.add_argument("-llmweight", "--LLM_edgeweight", help = "Whether to use the LLM generated edge weights", default="no")
    parser.add_argument("-pplweight", "--PPL_edgeweight", help = "Whether to use the Perplexity based edge weights", default="no")
    parser.add_argument("-nrcweight", "--NRC_edgeweight", help = "Whether to use the NRC Lexicon based edge weights", default="no")

    # Read arguments from command line
    args = parser.parse_args()
    
    # Working dir
    working_dir = "experiments_sst/"
    
    setting = args.setting
    num_iters = args.num_iters
    reward_fpath = args.reward_file
    data_fpath = args.data_file
    llama_bool = args.llama
    mbert_mlm = args.mbert_mlm
    prob_reward = args.probreward
    algorithm = args.algorithm
    LLM_edgeweight = args.LLM_edgeweight
    PPL_edgeweight = args.PPL_edgeweight
    NRC_edgeweight = args.NRC_edgeweight

    path = working_dir+setting+"histplots"
    ensure_directory_exists(path)

    df = pd.read_csv(data_fpath)
    # df2 = pd.read_csv("/home/debrupd1/XCOPA/hate_test_predictions/SMAB/SMAB/GPT3.5_ALLINFO_edges.csv")
    total_edges = len(df)#+len(df2)

    
    # Take input 
    #data_reward = [json.loads(lines) for lines in open("/home/debrupd1/XCOPA/hate_test_predictions/new_arms_reward.json", "r")]
    data_reward = [eval(lines) for lines in open(reward_fpath, "r")]
    #data_reward = json.load(open(reward_fpath,"r"))
    #print(type(data_reward[0]))
    
    tup, hash_ = get_tup_hash(data_reward)
    print("length of hash and tuple are: ", len(hash_), len(tup))
    assert len(hash_) == len(tup)
    
#     for k,_ in hash_.items():
#         h_reward = 0
#         samples = hash_[k]
#         print(samples)
#         for j in range(len(samples)):
#             if h_reward < samples[j][2]:
#                 h_reward = samples[j][2]
#         highest_reward[k].append(h_reward)

    test(tup, hash_)
    
    bandit_setup = Bandit(K = len(hash_), 
                          N = num_iters, 
                          hash_ = hash_, 
                          gram_tuple = tup,
                          total_edges=total_edges,
                          setting=setting, 
                          llama_true=llama_bool,
                          mbert_mlm=mbert_mlm,
                          prob_reward=prob_reward,
                          algorithm=algorithm,
                          working_dir=working_dir,
                          LLM_edgeweight=LLM_edgeweight,
                          PPL_edgeweight=PPL_edgeweight,
                          NRC_edgeweight=NRC_edgeweight)
    
    write_global_sensitivity_initvalues(tup, bandit_setup, working_dir, setting)
    
    bandit_setup.run()
    print("Total regret is: ", bandit_setup.total_regret)
    
    write_global_sensitivity_finalvalues(tup, bandit_setup, working_dir, setting)
#     write_regret_per_iteration(tup, bandit_setup, working_dir, setting)
    write_regret_freq_localsens(tup, bandit_setup, working_dir, setting)
    