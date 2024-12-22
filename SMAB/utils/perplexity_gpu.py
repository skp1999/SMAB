from tqdm import tqdm
import json 
import evaluate
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

perplexity = evaluate.load("perplexity", module_type="metric")
device = 'cuda:0'
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def calculate_perplexity(all_mlm_edges):

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ppls = []
    for edge in all_mlm_edges:
        inputs = tokenizer(edge, return_tensors = "pt")
        inputs = inputs.to(device)
        loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
        ppl = torch.exp(loss).cpu().detach().numpy()
        ppls.append(ppl)

    ppls_inv = [(1/i) for i in ppls]
    res = np.array(ppls_inv)/np.sum(ppls_inv)

    return res

    #return np.array(perplexities_inv)/np.sum(perplexities_inv)

reward_fpath = "../data/rotten_tomatoes/data_new/arms_MLM_bert_base_uncased_validation.json"
data_reward = [eval(lines) for lines in open(reward_fpath, "r")]

for i, sample in enumerate(tqdm(data_reward)):
    edges = sample['meta']['samples']
    for edge in edges:
        perplexities = calculate_perplexity(edge[2])
        edge.append(perplexities.tolist())
    with open('../data/rotten_tomatoes/data_new/arms_MLM_bert_base_uncased_validation_ppls.json', "a") as fl:
        json.dump(sample, fl)
        fl.write("\n")
#     json.dump(data_reward, open('../data/rotten_tomatoes/arms_MLM_bert_base_uncased_validation_ppls.json','w'), indent=4)

# start_time = time.time()
# device = 'cuda:0'
# model = AutoModelForCausalLM.from_pretrained("gpt2")
# model = model.to(device)

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# texts = ["become a citizen of the federal states or leave! it's simple you have no rights as an illegal. you have no grounds to be demanding."*10]
# inputs = tokenizer(texts, return_tensors = "pt")
# inputs = inputs.to(device)
# loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
# ppl1 = torch.exp(loss).cpu().detach().numpy()
# print(ppl1)

# texts = ["become a citizen of the united states or leave! it's simple you have no rights as an illegal. you have no grounds to be demanding."*10]
# inputs = tokenizer(texts, return_tensors = "pt")
# inputs = inputs.to(device)
# loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
# ppl2 = torch.exp(loss).cpu().detach().numpy()
# print(ppl2)
# end_time = time.time()

# print(end_time - start_time)