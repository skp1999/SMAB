from tqdm import tqdm
import json 
import evaluate
import numpy as np

perplexity = evaluate.load("perplexity", module_type="metric")
def calculate_perplexity(all_mlm_edges):
    
    results = perplexity.compute(model_id='gpt2',
                                 add_start_token=False,
                                 predictions=all_mlm_edges)
    perplexities = np.array(results["perplexities"])
    perplexities_inv = [(1/i) for i in perplexities]
    return np.array(perplexities_inv)/np.sum(perplexities_inv)

reward_fpath = "data/arms_MLM_mBERT_withprobs.json"
data_reward = [eval(lines) for lines in open(reward_fpath, "r")]

for i, sample in enumerate(tqdm(data_reward)):
    edges = sample['meta']['samples']
    for edge in edges:
        perplexities = calculate_perplexity(edge[2])
        edge.append(perplexities.tolist())
    json.dump(data_reward, open('data/arms_MLM_mBERT_withprobs_ppls.json','w'), indent=4)