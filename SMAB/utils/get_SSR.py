import numpy as np
import pandas as pd
import json
import re
import string
from collections import defaultdict, OrderedDict
from nltk.tokenize import wordpunct_tokenize, TweetTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
tweet_tokenizer = TweetTokenizer()
from transformers import pipeline
import torch
from tqdm import tqdm
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

class eng_mab_data:

    def __init__(self, df, arms, K):

        self.K = K
        self.df = df
        self.arms = arms
        self.hash_map = defaultdict(list)

        self.regex_digit = "\d"
        self.space_pattern = '\s+'
        self.giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_regex = '@\s[\w\-]+'

        self.stop_words_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        # except "no", "nor", "not"

        self.alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.prepare_edges()

    
    def only_punctuation(self, input_string):
        pattern = r'^[' + re.escape(string.punctuation) + r']+$'
        return bool(re.match(pattern, input_string))

    def check_word(self, W):

        if re.search(self.regex_digit, W) is None and not self.only_punctuation(W):
            # Strings/Words that don't contain any digit and are not composed of only punctuation symbol(s) will pass
            # Strings like "!@#$%^&" will not pass while strings like "sachin's" will pass although "sachin's" contain a punctuation symbol
            # Strings like "sac549hi6n" and "sachin4903" will not pass as they contain digits.

            return True
        else:
            return False

    def prepare_edges(self):
        
        if self.K == 1:
            for i in range(len(self.df)):
                sentence = self.df.iloc[i]['Text']
                parsed_text = re.sub(self.giant_url_regex, '', sentence)
                parsed_text = re.sub(self.mention_regex, '', parsed_text)
                parsed_text = re.sub(r"\b(RT|rt|fav|FAV)\b", '', parsed_text)
                parsed_text = re.sub(self.space_pattern, ' ', parsed_text)
                final_sent = tweet_tokenizer.tokenize(parsed_text)
                #print(final_sent)
                for elem in final_sent:
                    if "#" not in elem:
                        tokens = wordpunct_tokenize(elem)
                        #print(tokens)
                        for t in tokens:
                            if t in self.arms and t not in self.alphabets and t not in self.stop_words_list and self.check_word(t):
                                self.hash_map[t].append(i) # (original word, sentence id)
                    else:
                        if elem in self.arms and elem not in self.alphabets and elem not in self.stop_words_list and self.check_word(elem):
                                self.hash_map[elem].append(i) # (original word, sentence id)

            self.hash_map = dict(OrderedDict(sorted(self.hash_map.items(), key = lambda x : len(list(set(x[1]))))))
            print(len(self.hash_map))
            print("max length: ", len(list(self.hash_map.values())[-1])) 
            #self.create_dict()
        return self.hash_map

    def create_dict(self):
        dict_final = {}
        c = 0
        for k, v in self.hash_map.items():
            temp_dict = {}
            temp_dict['word'] = k
            temp_dict['sentence_ids'] = v
            dict_final['id'] = (c + 1)
            dict_final['meta'] = temp_dict
            c += 1
            #self.write_file(dict_final, "../data/rotten_tomatoes/eng_val_new_hash_map.json")

    def write_file(self, d, file_name):
        with open(file_name, "a") as fl:
            json.dump(d, fl)
            fl.write("\n")
            

class bert_predict_mlm():

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

class bert_predict_rt():

    def __init__(self):
        
        self.device = 'cuda:0'

        print("Loading Tokenizer")
        self.loaded_tokenizer = BertTokenizer.from_pretrained('textattack/bert-base-uncased-rotten-tomatoes')

        print("Loading Model")    
        self.loaded_model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-rotten-tomatoes", 
                                                                          num_labels = 2, 
                                                                          output_attentions = False, 
                                                                          output_hidden_states = False,)
        self.loaded_model = self.loaded_model.to(self.device)


    def run_eval_samples(self, sentences):
        MAX=512
        input_ids = []
        attention_masks = []
        for s in sentences:
            encoded_dict = self.loaded_tokenizer.encode_plus(
                s,                      # Sentence to encode.
                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                max_length = MAX,           # Pad & truncate all sentences.
                pad_to_max_length = True,
                return_attention_mask = True,   # Construct attn. masks.
                return_tensors = 'pt',     # Return pytorch tensors.
                   )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        
        final_pred_label_list = []
        probabilities_list = []

        #total_val_acc = 0
        for batch_idx, (Input_ids, mask_ids) in enumerate(tuple(zip(input_ids, attention_masks))):
            Input_ids = Input_ids.to(self.device)
            mask_ids = mask_ids.to(self.device)
        
            with torch.no_grad():
                prediction = self.loaded_model(Input_ids, 
                                            token_type_ids = None, 
                                            attention_mask = mask_ids).logits

            prediction = prediction.detach().cpu()
            final_pred_label_list.extend(torch.log_softmax(prediction, dim = 1).argmax(dim = 1).tolist())
            probabilities_list.extend((torch.log_softmax(prediction, dim = 1)).tolist())
        
        return final_pred_label_list, probabilities_list

    
def calculate_SSR(data, df):
    word_level_SSR_cnt = 0
    sent_level_SSR_cnt = [False]*len(df)
    for dat in tqdm(data):
        flag=False
        for sample in dat['meta']:
            if(sample['flip']==True):
                if(sent_level_SSR_cnt[sample['idx']]==False):
                    sent_level_SSR_cnt[sample['idx']]=True                 
                flag=True
                break
        if(flag==True):
            word_level_SSR_cnt+=1

    word_level_SSR = word_level_SSR_cnt/len(data)
    sent_level_SSR = sum(sent_level_SSR_cnt)/len(df)
    print(f'Word level SSR: {word_level_SSR}')
    print(f'Sentence level SSR: {sent_level_SSR}')

if __name__ == "__main__":
    
    # Read sensitivity final values
    global_sens_final_fpath = "../experiments_new/thompson_01_eg_ppl_rt_400k/global_sensitivity_finalvalues.json"
    global_sens_final_data = [eval(lines) for lines in open(global_sens_final_fpath,'r')]
    global_sens_word_gt_thres = []
    threshold = 0.7
    for data in global_sens_final_data:
        for key, value in data.items():
            if(value>threshold):
                global_sens_word_gt_thres.append(key)
    print(f'Words with sens greater than {threshold} : {len(global_sens_word_gt_thres)}')
        
    
    # Read test generated csv
    test_df = pd.read_csv('../data/rotten_tomatoes/data_test/generated_text_0308.csv')
    words_with_sent_ids = eng_mab_data(test_df, global_sens_word_gt_thres, 1).hash_map
    
    
    #for each arm -> loop through the sentences and get the MLMs, if any of it flips the label (Success)
    mlm_obj = bert_predict_mlm()
    cls_obj = bert_predict_rt()
    all_words_flips = []
    for word, sent_ids in tqdm(words_with_sent_ids.items()):
        sent_flips = []
        for idx in sent_ids:
            #print(idx)
            sentence = test_df.iloc[idx]['Text']
            mlm_output_dict = mlm_obj.get_mbert_mlm_output(sentence, word)
            mlm_sentences = mlm_output_dict['mlm_sentences']
            sentence_pred_label, _ = cls_obj.run_eval_samples([sentence])
            mlm_pred_labels, _ = cls_obj.run_eval_samples(mlm_sentences)
            flip = False
            for label in mlm_pred_labels:
                if sentence_pred_label[0] != label:
                    flip=True
                    break
            sent_flips.append({'idx': idx, 'flip': flip})
        all_words_flips.append({'word': word, 'meta': sent_flips})
#         with open('../data/rotten_tomatoes/data_new/all_words_flips_0.9.json', "a") as fl:
#             json.dump(d, fl)
#             fl.write("\n")
                    
        json.dump(all_words_flips, open('../data/rotten_tomatoes/data_new/all_words_flips_0.7.json','w'), indent=4)      
     
    #Report the SSR
    calculate_SSR(all_words_flips, test_df)        