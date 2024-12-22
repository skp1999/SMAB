import re
import csv
import json
import string
import pandas as pd
from collections import defaultdict, OrderedDict
from nltk.tokenize import wordpunct_tokenize, TweetTokenizer
import random

tweet_tokenizer = TweetTokenizer()

class eng_mab_data:

    def __init__(self, df, K, output_file):

        self.K = K
        self.df = df
        #self.arms = arms
        self.hash_map = defaultdict(list)
        self.output_file = output_file

        self.regex_digit = "\d"
        self.space_pattern = '\s+'
        self.giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_regex = '@\s[\w\-]+'

        self.stop_words_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
        

        self.alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.prepare_edges()
        #self.create_batches()

    
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
                sentence = self.df.iloc[i]['Text'].lower()
                
#                 sentence1 = self.df.iloc[i]['sentence1']
#                 sentence2 = self.df.iloc[i]['sentence2']
#                 sentence = sentence1.strip() + " " + sentence2.strip()
                
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
                            if t not in self.alphabets and t not in self.stop_words_list and self.check_word(t):
                                self.hash_map[t].append(i) # (original word, sentence id)
                    else:
                        if elem not in self.alphabets and elem not in self.stop_words_list and self.check_word(elem):
                                self.hash_map[elem].append(i) # (original word, sentence id)

            self.hash_map = dict(OrderedDict(sorted(self.hash_map.items(), key = lambda x : len(list(set(x[1]))))))
            print(len(self.hash_map))
            print("max length: ", len(list(self.hash_map.values())[-1]))
            self.create_dict()

    def create_dict(self):
        dict_final = {}
        c = 0
        for k, v in self.hash_map.items():
            temp_dict = {}
            temp_dict['word'] = k
            if(len(v)>20):
                temp_dict['sentence_ids'] = random.sample(v, 20)
            else:
                temp_dict['sentence_ids'] = v
            dict_final['id'] = (c + 1)
            dict_final['meta'] = temp_dict
            c += 1
            self.write_file(dict_final, self.output_file)

    def write_file(self, d, file_name):
        with open(file_name, "a") as fl:
            json.dump(d, fl)
            fl.write("\n")


folder_path = "../data/dataset_sst"
data = pd.read_csv(f"{folder_path}/validation.csv")
output_file = f"{folder_path}/arms_hashmap.json"

# data_arms = [eval(lines) for lines in open('../data/rotten_tomatoes/eng_val_hash_map.json','r')]
# arms = []
# for arm_info in data_arms:
#     arms.append(arm_info['meta']['word'])
mab_dataprep = eng_mab_data(data, 1, output_file)