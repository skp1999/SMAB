import os
import csv
import json
import torch
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import statistics
from pdb import set_trace as bp



def csv_to_json(ip_json_file, op_json_file, csv_file):
    
    data_reward = [eval(lines) for lines in open(ip_json_file, "r")]
    df = pd.read_csv(csv_file)
    new_arms = []
    
    with open(op_json_file, 'w') as file:
        
        for i in tqdm(range(len(data_reward))):

            try:
                arm = data_reward[i]['meta']['word']

                # Samples is the list of edges
                samples = data_reward[i]['meta']['samples']
                samples_new = []
                
                for sample in samples:
                    example_no = sample[0]
                    replacement_words = sample[1]
                    sentences = sample[2]
                    mlm_scores = sample[3]

                    new_samples = []

                    # Find in the dataframe df with arm - multiple edges 
                    armmatch_df = df[(df['arm'] == arm)]


                    if len(armmatch_df) == 0:
                        new_arms.append(data_reward[i]) # Keep as before 
                        continue

                    armmatch_df = armmatch_df.reset_index(drop=True)

                    #print("Before Adding column",armmatch_df)
                    probs = armmatch_df['probabilities'].tolist()
                    preds = armmatch_df["prediction"].tolist()

                    new_probs = []
                    new_preds = []
                    bool_original = armmatch_df["bool_original"].tolist()


                    for j in range(len(replacement_words)):
                        index_of_value = armmatch_df.index[armmatch_df['new_word'] == replacement_words[j]].tolist()
                        new_preds.append(preds[index_of_value[0]])
                        new_probs.append(probs[index_of_value[0]])

                    orig_ind = (armmatch_df.index[armmatch_df['bool_original'] == 1].tolist())[0]
                    orig_pred = preds[orig_ind]
                    orig_prob = probs[orig_ind]

                    pred_list = [orig_pred]
                    pred_list.append(new_preds)

                    prob_list = [orig_prob]
                    prob_list.append(new_probs)

                    sample.append(pred_list) 
                    sample.append(prob_list)
                    samples_new.append(sample)

                data_reward[i]['meta']['samples'] = samples_new
                new_arms.append(data_reward[i])
                s = str(data_reward[i])
                file.write(s)
                file.write('\n')
            except:
                continue


def append_to_csv(csv_file, data, headers=None):
 
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_file)

    # Open the file in append mode
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # If the file does not exist, write the headers first
        if not file_exists:
            writer.writerow(headers)

        # Transpose the lists to rows and write the data
        writer.writerows(data)

    #print(f"Data has been appended to {csv_file}")



if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")



class load_and_eval:

    def __init__(self, df_testing, model_path, num_labels, out_csv):
        
        self.model_path = model_path
        self.num_labels = num_labels
        self.out_csv = out_csv
        self.df_testing = df_testing
        
#         print("Loading Tokenizer")
#         self.loaded_tokenizer = BertTokenizer.from_pretrained('../data/eng/eng/')

#         print("Loading Model")    
#         self.loaded_model = BertForSequenceClassification.from_pretrained("../data/eng/eng/", 
#                                                                           num_labels = 2, 
#                                                                           output_attentions = False, 
#                                                                           output_hidden_states = False,
#                                                                           ignore_mismatched_sizes=True)
        print("Loading Tokenizer")
        self.loaded_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        print("Loading Model")  
        self.loaded_model = DistilBertForSequenceClassification.from_pretrained(model_path,
                                                                                num_labels = self.num_labels,
                                                                                output_attentions = False,
                                                                                output_hidden_states = False,)

        
#         self.loaded_tokenizer = BertTokenizer.from_pretrained(model_path)
#         self.loaded_model = BertForSequenceClassification.from_pretrained(self.model_path, 
#                                                                              num_labels = self.num_labels, 
#                                                                              output_attentions = False, 
#                                                                              output_hidden_states = False,)
        
        self.loaded_model = self.loaded_model.to(device)

        self.test_data = self.prepare_data(self.df_testing, 512)
        self.test_load = self.prepare_data_loader()
        print('####### DataLoader prepared ##########')
        self.df_output = self.eval()

    def prepare_data(self, df_testing, MAX):
        
        input_ids = []
        attention_masks = []

        sent = []
        #labels = []
        
        # Input for CSV file 
        
        for i in tqdm(range(len(df))):
            sent.append(df.iloc[i]['text'])
            #labels.append(df.iloc[i]['label'])
        
        

        for s in sent:
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
        input_ids = torch.cat(input_ids, dim = 0)
        attention_masks = torch.cat(attention_masks, dim = 0)
        #labels = torch.tensor(labels)

        #dataset = TensorDataset(input_ids, attention_masks, labels)
        dataset = TensorDataset(input_ids, attention_masks)
        return dataset

    def prepare_data_loader(self, batch_size = 32, shuffle = False):
        test_loader = DataLoader(self.test_data, shuffle = shuffle, batch_size = batch_size)
        return test_loader

    def multi_acc(self, y_pred, y_test):
        acc = (torch.log_softmax(y_pred, dim = 1).argmax(dim = 1) == y_test).sum().float() / float(y_test.size(0))
        return acc

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
        
    def eval(self):
        final_pred_label_list = []
        probabilities_list = []
        #total_val_acc = 0
        print(f'#### Length of data loaded: {len(self.test_load)} #####')
        for batch_idx, (Input_ids, mask_ids) in tqdm(enumerate(self.test_load)):
            Input_ids = Input_ids.to(device)
            mask_ids = mask_ids.to(device)
            #labels = y.to(device)
        
            with torch.no_grad():
                '''
                loss, prediction = self.loaded_model(Input_ids, 
                                              token_type_ids = None, 
                                              attention_mask = mask_ids, 
                                              labels = labels).values()
                '''      


                prediction = self.loaded_model(Input_ids).logits


            #print(type(prediction))
            #print(prediction)
            #print(prediction.shape)
            prediction = prediction.detach().cpu()
            #print("One pred done!")
            #labels = labels.detach().cpu()
            final_pred_label_list.extend(torch.log_softmax(prediction, dim = 1).argmax(dim = 1).tolist())
            probabilities_list.extend((torch.softmax(prediction, dim = 1)).tolist())
            #acc = self.flat_accuracy(prediction.numpy(), labels.numpy())
            #acc = self.multi_acc(prediction, labels)
            #total_val_acc  += acc.item()
        #print("evaluation accuracy is: ", total_val_acc/ len(self.test_load))
        
        df = self.df_testing
        df["prediction"] = final_pred_label_list
        df["probabilities"] = probabilities_list
        
        df.to_csv(self.out_csv,index=False)
        
        print("length of pred label list is: ", len(final_pred_label_list))
        
        return df


def replace_word(sentence,index,new_word):
    
    #Find the end index of the current word starting at the given index
    
    start_index = index
    end_index = sentence.find(" ", start_index)

    if end_index == -1:  # If there's no space, the word goes until the end of the string
        end_index = len(sentence)

    # Extract parts of the string
    before_word = sentence[:start_index]
    after_word = sentence[end_index:]

    # Reconstruct the string with the new word
    new_sentence = before_word + new_word + after_word
    
    return new_sentence



def formulate_csv(json_file, csv_fname):
    
    json_input = [eval(lines) for lines in open(json_file, "r")]
    headers = ["arm","example_id","new_word","mlm_score", "text","bool_original"]
    
    data = pd.DataFrame(columns=headers)
    
    for i in tqdm(range(len(json_input))):  #len(json_input)
        
        # ith Arm
        arm = json_input[i]['meta']['word']

        # Samples is the list of edges
        print("Len samples: ",len(json_input[i]['meta']['samples']))
        samples = json_input[i]['meta']['samples']
        
        for sample in samples:
            #print(sample)
            example_no = sample[0]
            replacement_words = sample[1]
            sentences = sample[2]
            mlm_scores = sample[3]

            indices = []

            for i in range(len(replacement_words)):
                word = replacement_words[i].strip()
                index = sentences[i].find(word)
                indices.append(index)

            ind = statistics.mode(indices)

            original_sent = replace_word(sentences[0],ind,arm)

            '''
            print(arm)
            print(example_no)
            print(replacement_words)
            print(sentences)
            print(mlm_scores)
            '''

            arms = [arm]*(len(replacement_words)+1)
            example_ids = [example_no]*(len(replacement_words)+1)

            bool_original = [0]*len(replacement_words)

            bool_original.append(1)
            replacement_words.append(arm)
            sentences.append(original_sent)
            mlm_scores.append(-1)

            data = zip(arms, example_ids, replacement_words, mlm_scores, sentences, bool_original)
            try:
                append_to_csv(csv_fname, data, headers=headers)
            except:
                continue
    


if __name__ == "__main__":
    
    model_path = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"  # Set the model path here 
    
    folder_path = "../data/dataset_sst"
    
    df = formulate_csv(json_file = f"{folder_path}/arms_MLM_xlmr_validation.json",
                      csv_fname = f"{folder_path}/temp.csv")
    df = pd.read_csv(f"{folder_path}/temp.csv")
    
    # Get predictions
    print('##### Evaluating ######')
    obj = load_and_eval(df_testing= df, model_path = model_path, num_labels = 2, out_csv = f"{folder_path}/temp_preds.csv")
    print('##### Predictions Done #####')
   
    
    # Convert csv file to json 
    csv_to_json(ip_json_file=f"{folder_path}/arms_MLM_xlmr_validation.json",
               op_json_file=f"{folder_path}/arms_MLM_xlmr_validation_preds.json",
                csv_file=f"{folder_path}/temp_preds.csv")