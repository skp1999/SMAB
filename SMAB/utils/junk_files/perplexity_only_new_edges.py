import re
import csv
import math
import json
import torch
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from collections import defaultdict, OrderedDict
from transformers import BertTokenizer, AutoTokenizer, BertForSequenceClassification, XLMRobertaForSequenceClassification
# from run_bert_RT import bert_predict_rt

import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer

perplexity = evaluate.load("perplexity", module_type="metric")
device = 'cuda:0'
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
        
def is_ppl_available(word, sent_id):
    for data in datas:
        if(data['meta']['word']==word):
            samples = data['meta']['samples']
            for sample in samples:
                if(sample[0]==sent_id):
                    return True, data['meta']
                
    return False, []



def calculate_perplexity(all_mlm_edges):
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



if __name__ == "__main__":
    print("running perplexity only on new edges..")
    tokenizer_kwargs = {"truncation": True}

    org_file = pd.read_csv("../data/rotten_tomatoes/data_new/validation_new.csv")
    
    
    data_arms = [eval(lines) for lines in open('../data/rotten_tomatoes/data_new/eng_val_new_hash_map.json','r')]
    data_mlms = [eval(lines) for lines in open('../data/rotten_tomatoes/data_new/arms_MLM_bert_base_uncased_validation.json','r')]
    datas = [eval(lines) for lines in open('../data/rotten_tomatoes/arms_MLM_bert_base_uncased_validation_ppls.json','r')]
   
    print('Start adding')
    
    available, not_available = 0, 0
    
    for _, data in enumerate(tqdm(data_arms)):
        idx = data['id']
        word = data['meta']['word']
        sent_ids = data['meta']['sentence_ids']

        for sent_id in sent_ids:
            ppl_available, ppl = is_ppl_available(word, sent_id)
            org_text = org_file.iloc[sent_id]['Text']

            if(ppl_available==True):
                available+=1
#                 d = {}
#                 d['id'] = idx
#                 d['meta'] = ppl
#                 print(d)
#                 with open('../data/rotten_tomatoes/data_new/arms_MLM_bert_base_uncased_validation_ppls_new.json', "a") as fl:
#                     json.dump(d, fl)
#                     fl.write("\n")
#                 fl.close()

            elif(ppl_available==False):
                not_available+=1
                #if(word==data_mlms[data['id']]['meta']['word']):
                edges = data_mlms[data['id']]['meta']['samples']
                for edge in edges:
                        if(edge[0]==sent_id):
                            perplexities = calculate_perplexity(edge[2])
                            edge.append(perplexities.tolist())
                            d = {}
                            d['id'] = idx
                            d['meta'] = data_mlms[data['id']]['meta']
                            print(d)
                            with open('../data/rotten_tomatoes/data_new/arms_MLM_bert_base_uncased_validation_ppls_new.json', "a") as fl:
                                json.dump(d, fl)
                                fl.write("\n")
                            fl.close()
               
                
        print(available, not_available)

                        
#         except:
#             print(data['id'])
#             continue
        
    
#     for arm in tqdm(arms_file):
#         dict_word_samples = defaultdict(list)
#         f_list = masking_input_prep(org_file, arm)
#         arm['meta']['samples'] = []
#         masked_sent_list, masked_word_list, masked_conf_list = [], [], []
#         for i in range(len(f_list)):
#             sent_id = f_list[i][0]
#             word = f_list[i][1]
#             sent = f_list[i][2]
#             try:
#                 assert word in sent
#             except:
#                 print(word)
#                 print(sent)
#                 raise AssertionError("word is not present in the sentence.")

#             new_sent = sent.replace(word, "[MASK]", 1)
#             if not new_sent.endswith("."):
#                 new_sent = new_sent + "."

#             output_xlmr = fill_masker(new_sent)

#             '''
#             except:
#                 print(word)
#                 print(batch['meta']['batch_data'][i][1])
#                 print("******************************************************************")
#                 Phrase = select_words_before(new_sent, "<mask>", 8)

#                 assert Phrase in new_sent
#                 parts = new_sent.split(Phrase.strip())
#                 assert len(parts) == 2
#                 assert "<mask>" in parts[1].strip()

#                 output_xlmr = fill_masker(Phrase + " " + parts[1].strip())
#                 for K in range(len(output_xlmr)):
#                     output_xlmr[K]['token_str'] = parts[0].strip() + " " + output_xlmr[K]['token_str']
#              '''       
#             for j in range(len(output_xlmr)):
#                 masked_word_list.append(output_xlmr[j]['token_str'])

#             for j in range(len(output_xlmr)):
#                 masked_conf_list.append(output_xlmr[j]['score'])

#             for j in range(len(output_xlmr)):
#                 masked_sent_list.append(output_xlmr[j]['sequence'])

            
#             arm['meta']['samples'].append([sent_id, masked_word_list, masked_sent_list, masked_conf_list])
            
          
#         with open('../data/rotten_tomatoes/data_new/arms_MLM_bert_base_uncased_validation_new.json', "a") as fl:
#             json.dump(arm, fl)
#             fl.write("\n")
