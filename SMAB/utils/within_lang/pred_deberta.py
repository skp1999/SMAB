import pandas as pd
import os
import json
from tqdm import tqdm 
import numpy as np
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v1")


def get_preds_rewards_file():
    df = pd.read_csv('../../data/dataset_mhate/english/new/valid.csv')
    rewards_file = [eval(line) for line in open('../../data/dataset_mhate/english/new/xlmr_reward_withoutgold_mbert_eng.json','r')]
    for reward in tqdm(rewards_file):
        word = reward['meta']['word']
        samples = reward['meta']['samples']
        for sample in samples:
            del sample[2]
            idx = sample[0]
            mlm_words = sample[1]
            sent = df.iloc[idx]['Text'].strip()
            preds = []
            sequence_to_classify = sent
            candidate_labels = ["hateful", "non-hateful"]
            cls = pipe(sequence_to_classify, candidate_labels, multi_label=False)
            idx = np.argmax(cls['scores'])
            label = candidate_labels[idx]
#             print(cls)
            if "non-hateful" in label.lower():
                preds.append(0)
            else:
                preds.append(1)
            preds.append(list())
            for word1 in mlm_words:
                sent1 = sent.replace(word, word1)
                cls = pipe(sent1, candidate_labels, multi_label=False)
                idx = np.argmax(cls['scores'])
                label = candidate_labels[idx]
                #print(cls)
                if "non-hateful" in label.lower():
                    preds[1].append(0)
                else:
                    preds[1].append(1)
            sample.append(preds)
        with open('../../data/dataset_mhate/english/new/xlmr_reward_withoutgold_deberta_eng_updated.json','a') as f:
            json.dump(reward,f)
            f.write('\n')
            
def get_acc_test_file():
    df = pd.read_csv('../../data/dataset_mhate/english/new/valid.csv')
    preds = []
    for index, row in tqdm(df.iterrows()):
        sent = row['Text']
        sequence_to_classify = sent
        candidate_labels = ["hateful", "non-hateful"]
        cls = pipe(sequence_to_classify, candidate_labels, multi_label=False)
        idx = np.argmax(cls['scores'])
        label = candidate_labels[idx]
        #print(cls, label)
        if "non-hateful" in label.lower():
            preds.append(0)
        else:
            preds.append(1)
    df['pred'] = preds
    df.to_csv('../../data/dataset_mhate/english/new/valid_preds_deberta.csv', index=False)
    
get_preds_rewards_file()
    
                
                
                
                
