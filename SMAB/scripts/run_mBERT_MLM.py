
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)


class mBERT_predict_mlm():

    def __init__(self):
        
        self.device = 'cuda:0'
        self.unmasker = pipeline('fill-mask', 
                            model='bert-base-uncased',
                            top_k = 10, 
                            device_map=self.device)
        print("mBERT MLM pipeline initialized!")    
         
    def get_mbert_mlm_output(self, sentence, word):
        input_text = sentence.replace(word, "[MASK]",1)
        #print(sentence)
        #print(input_text)

        output_dict = dict()
        output_dict['word'] = word
        output_dict['generated_sentence'] = sentence

        output_dict['mlm_words'] = list()
        output_dict['mlm_sentences'] = list()
        output_dict['mlm_scores'] = list()

        with torch.no_grad():
            # Perform inference on the batch
            batch_results = self.unmasker(input_text)
            #print(batch_results)


            for j in range(len(batch_results)):

                result_edge  = batch_results[j]
                #print(result_edge)

                for w in [result_edge]:
                    output_dict['mlm_words'].append(w['token_str'])
                    output_dict['mlm_sentences'].append(w['sequence'])
                    output_dict['mlm_scores'].append(w['score'])

        return output_dict