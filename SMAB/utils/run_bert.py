import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
import random
import os
import pandas as pd
import json

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

class bert_predict_rt():

    def __init__(self):
        
        self.device = 'cuda:0'

        print("Loading Tokenizer")
        self.loaded_tokenizer = BertTokenizer.from_pretrained('../data/eng/eng/')

        print("Loading Model")    
        self.loaded_model = BertForSequenceClassification.from_pretrained("../data/eng/eng/", 
                                                                          num_labels = 3, 
                                                                          output_attentions = False, 
                                                                          output_hidden_states = False,
                                                                          ignore_mismatched_sizes=True)
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
        for batch_idx, (Input_ids, mask_ids) in tqdm(enumerate(tuple(zip(input_ids, attention_masks)))):
            Input_ids = Input_ids.to(self.device)
            mask_ids = mask_ids.to(self.device)
        
            with torch.no_grad():
                prediction = self.loaded_model(Input_ids, 
                                            token_type_ids = None, 
                                            attention_mask = mask_ids).logits

            prediction = prediction.detach().cpu()
            final_pred_label_list.extend(torch.log_softmax(prediction, dim = 1).argmax(dim = 1).tolist())
            #probabilities_list.extend((torch.log_softmax(prediction, dim = 1)).tolist())
        
        return final_pred_label_list, probabilities_list

device = 'cuda:0'

class load_and_eval:

    def __init__(self, df_testing, model_path, num_labels, out_csv):
        
        self.model_path = model_path
        self.num_labels = num_labels
        self.out_csv = out_csv
        self.df_testing = df_testing
        print("Loading Tokenizer")
        self.loaded_tokenizer = BertTokenizer.from_pretrained(self.model_path)

        print("Loading Model")    
        self.loaded_model = BertForSequenceClassification.from_pretrained(self.model_path, 
                                                                          num_labels = 3, 
                                                                          output_attentions = False, 
                                                                          output_hidden_states = False,)
        
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
            sent.append(df.iloc[i]['Text'])
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


                prediction = self.loaded_model(Input_ids, 
                                            token_type_ids = None, 
                                            attention_mask = mask_ids).logits


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
    
    
obj = bert_predict_rt()
df = pd.read_csv('../data/hate/hate/dev_hate.csv')
f = open('../data/hate/hate/arms_MLM_mbert_base_cased_val_hate.json' ,'r')
# print(len(f.readlines()))
for i, line in tqdm(enumerate(f)):
    if 1:
        arm = eval(line)
        edges = arm['meta']['samples']
        for edge in edges:
            sent = df.iloc[edge[0]]['Text']
            sent_pred = obj.run_eval_samples([sent])[0][0]
            mlm_edges = edge[2]
            bert_preds, bert_probs = obj.run_eval_samples(mlm_edges)
            edge.insert(4,[sent_pred, bert_preds])
            
        with open('../data/hate/hate/arms_MLM_mbert_base_cased_val_hate_preds.json','a') as f1:
            json.dump(arm, f1)
            f1.write('\n')
#     except:
#         print(i)
#         continue


# model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# folder_path = '../data/checklist/checklist_some_types_500'
# obj = bert_predict_rt()
# df = pd.read_csv(f'{folder_path}/checklist_some_types_500.csv')
# load_and_eval(df_testing= df, model_path = model_path, num_labels = 2, out_csv = f"{folder_path}/checklist_some_types_500_preds.csv")

                                         

# texts = df['Text'].tolist()
# pred_labels, probs = obj.run_eval_samples(texts)
# df['pred_label'] = pred_labels
# df['probs'] = probs
# print(df)
# df.to_csv('../data/checklist/checklist_some_types_500/checklist_some_types_500_preds.csv', index=False)
############

# obj = bert_predict_rt()
# sample = ["@united what time does check in open for flight no UA80 from Manchester to Newark today ?. That's great!", "@united what time does check in open for flight no UA80 from Manchester to Newark today ?"]
# label, _ = obj.run_eval_samples(sample)
# print(label)