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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="google-bert/bert-base-uncased")
parser.add_argument("--folder_path", help="display a square of a given number")
parser.add_argument("--csv_file", help="display a square of a given number")
parser.add_argument("--arms_file", help="display a square of a given number")
parser.add_argument("--output_file", help="display a square of a given number")

args = parser.parse_args()


def write_file(dict_name, file_name):
    with open(file_name, "a") as fl:
        json.dump(dict_name, fl)
        fl.write("\n")

def masking_input_prep(data_file, arm):
    arms_list = []
    arm_word = arm['meta']['word']
    arm_word_edges = arm['meta']['sentence_ids']
    for i in range(len(arm_word_edges)):
        arms_list.append([arm_word_edges[i], arm_word, data_file.iloc[arm_word_edges[i]]['Text']])
    return arms_list


if __name__ == "__main__":
    print("running MLM ...")
    
    model_name = args.model_name
    folder_path = args.folder_path
    csv_filename = args.csv_file
    arms_filename = args.arms_file
    output_filename = args.output_file
    
    print(folder_path)

    org_file = pd.read_csv(f"{folder_path}/{csv_filename}")
    arms_file = [json.loads(lines) for lines in open(f"{folder_path}/{arms_filename}", "r")]
    
    tokenizer_kwargs = {"truncation": True}
    fill_masker = pipeline('fill-mask', model = model_name, top_k = 10, device = 0)
    
    arms_data_new = []
    print("There are a total of {} arms".format(len(arms_file)))
    
    for arm in tqdm(arms_file):
        dict_word_samples = defaultdict(list)
        f_list = masking_input_prep(org_file, arm)
        arm['meta']['samples'] = []
        
        for i in range(len(f_list)):
            masked_sent_list, masked_word_list, masked_conf_list = [], [], []
            sent_id = f_list[i][0]
            word = f_list[i][1]
            sent = f_list[i][2].lower()
            try:
                assert word in sent
            except:
                print(word)
                print(sent)
                raise AssertionError("word is not present in the sentence.")

            new_sent = sent.replace(word, "<mask>", 1)
            if not new_sent.endswith("."):
                new_sent = new_sent + "."

            output_xlmr = fill_masker(new_sent)

            '''
            except:
                print(word)
                print(batch['meta']['batch_data'][i][1])
                print("******************************************************************")
                Phrase = select_words_before(new_sent, "<mask>", 8)

                assert Phrase in new_sent
                parts = new_sent.split(Phrase.strip())
                assert len(parts) == 2
                assert "<mask>" in parts[1].strip()

                output_xlmr = fill_masker(Phrase + " " + parts[1].strip())
                for K in range(len(output_xlmr)):
                    output_xlmr[K]['token_str'] = parts[0].strip() + " " + output_xlmr[K]['token_str']
             '''       
            for j in range(len(output_xlmr)):
                masked_word_list.append(output_xlmr[j]['token_str'])

            for j in range(len(output_xlmr)):
                masked_conf_list.append(output_xlmr[j]['score'])

            for j in range(len(output_xlmr)):
                masked_sent_list.append(output_xlmr[j]['sequence'])

            
            arm['meta']['samples'].append([sent_id, masked_word_list, masked_sent_list, masked_conf_list])
            
          
        with open(f'{folder_path}/{output_filename}', "a") as fl:
            json.dump(arm, fl)
            fl.write("\n")