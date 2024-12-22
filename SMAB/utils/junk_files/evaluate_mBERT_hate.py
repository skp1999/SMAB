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
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

class mbertfinetuning:
    
    def __init__(self, train_loader, val_loader, test_loader, epochs, output_dir, no_labels, model = None, tokenizer = None):
        self.epochs = epochs
        self.training_stats = []
        self.no_labels = no_labels
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.output_dir = output_dir

        if tokenizer is None:
            print("tokenizer is None")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        else:
            print("tokenizer is not None")
            self.tokenizer = tokenizer
        if model is None:
            print("model is None")
            self.model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels = self.no_labels, output_attentions = False, output_hidden_states = False,)
        else:
            print("model is not None")
            self.model = model
        
        self.model = self.model.to(device)
        self.print_params(self.model)
        total_steps = len(self.train_loader) * self.epochs

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr = 2e-5, correct_bias = False)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
        self.bert_finetuning()
            
    def multi_acc(self, y_pred, y_test):
        acc = (torch.log_softmax(y_pred, dim = 1).argmax(dim = 1) == y_test).sum().float() / float(y_test.size(0))
        return acc


    def bert_finetuning(self):
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        

        for i in tqdm(range(self.epochs)):


            total_train_loss = 0
            total_train_acc  = 0
            self.model.train()
            
            for batch_idx, (Input_ids, mask_ids, y) in enumerate(self.train_loader):
                Input_ids = Input_ids.to(device)
                mask_ids = mask_ids.to(device)
                labels = y.to(device)

                self.optimizer.zero_grad()

                loss, prediction = self.model(Input_ids, 
                             token_type_ids=None,
                             attention_mask=mask_ids, 
                             labels=labels).values()

                acc = self.multi_acc(prediction, labels)
                total_train_loss += loss.item()
                total_train_acc  += acc.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            train_acc  = total_train_acc/len(self.train_loader)
            train_loss = total_train_loss/len(self.train_loader)

            self.model.eval()

            total_val_acc  = 0
            total_val_loss = 0

            for batch_idx, (Input_ids, mask_ids, y) in enumerate(self.val_loader):
                Input_ids = Input_ids.to(device)
                mask_ids = mask_ids.to(device)
                labels = y.to(device)

                with torch.no_grad():
                    loss, prediction = self.model(Input_ids, 
                             token_type_ids = None, 
                             attention_mask = mask_ids, 
                             labels = labels).values()
                prediction = prediction.detach().cpu()
                labels = labels.detach().cpu()
                acc = self.multi_acc(prediction, labels)
                total_val_loss += loss.item()
                total_val_acc  += acc.item()

            val_acc  = total_val_acc/len(self.val_loader)
            val_loss = total_val_loss/len(self.val_loader)
            
            self.training_stats.append({'epoch': i + 1, 'Training_accuracy': train_acc, 'Training_loss': train_loss, "Validation_accuracy": val_acc, "Validation_loss": val_loss})
        print("")
        print("Training complete!")
        
        self.save_model()
        self.plot_graph()


    def save_model(self):
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        print("Saving model to %s" % self.output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    def plot_graph(self):
        df_stats = pd.DataFrame(data = self.training_stats)
        df_stats = df_stats.set_index('epoch')
        print("**************************************")
        print(df_stats)
        df_stats.to_csv(self.output_dir + "training_stats.csv", header = True)
        print("**************************************")
        sns.set(style ='darkgrid')
        sns.set(font_scale = 1.5)
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.plot(df_stats['Training_loss'], 'b-o', label = "Training")
        plt.plot(df_stats['Validation_loss'], 'g-o', label = "Validation")
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3])
        plt.savefig(self.output_dir + "Loss_plot.jpg")
        
    def print_params(self, M):
        params = list(M.named_parameters())
        print('The BERT model has {:} different named parameters.\n'.format(len(params)))
        print('==== Embedding Layer ====\n')
        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== First Transformer ====\n')
        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== Output Layer ====\n')
        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    def evaluate(self):
        
        loaded_model = BertForSequenceClassification.from_pretrained(self.output_dir)
        loaded_tokenizer = BertTokenizer.from_pretrained(self.output_dir)
        loaded_model = loaded_model.to(device)
        self.print_params(loaded_model)

        total_val_acc = 0
        for batch_idx, (Input_ids, mask_ids, y) in enumerate(self.test_loader):
            Input_ids = Input_ids.to(device)
            mask_ids = mask_ids.to(device)
            labels = y.to(device)
        
            with torch.no_grad():
                loss, prediction = self.model(Input_ids, 
                                              token_type_ids = None, 
                                              attention_mask = mask_ids, 
                                              labels = labels).values()
            prediction = prediction.detach().cpu()
            labels = labels.detach().cpu()
            acc = self.multi_acc(prediction, labels)
            total_val_acc  += acc.item()
        print("evaluation accuracy is: ", total_val_acc/ len(self.test_loader))

class load_and_eval:

    def __init__(self, df_testing, out_dir = None):
        self.out_dir = out_dir
        self.df_testing = df_testing
        if self.out_dir is None:
            print("tokenizer is None")
            self.loaded_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        else:
            print("tokenizer is not None")
            self.loaded_tokenizer = BertTokenizer.from_pretrained(self.out_dir)

        if self.out_dir is None:
            print("model is None")
            self.loaded_model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels = 2, output_attentions = False, output_hidden_states = False,)
        else:
            print("model is not None")
            self.loaded_model = BertForSequenceClassification.from_pretrained(self.out_dir)
        
        self.loaded_model = self.loaded_model.to(device)

        self.test_data = self.prepare_data(self.df_testing, 512)
        self.test_load = self.prepare_data_loader()
        self.eval()

    def prepare_data(self, df, MAX):
        input_ids = []
        attention_masks = []

        sent = []
        #labels = []
        for i in range(len(df)):
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
            print("One pred done!")
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
        df.to_csv("/home/debrupd1/XCOPA/hate_test_predictions/mBERT_mlm_edges.csv",index=False)
        
        print("length of pred label list is: ", len(final_pred_label_list))



if __name__ == "__main__":
    
    local_dir = "/home/debrupd1/XCOPA/hate_test_predictions/mBERT_eng_only_hate"
    df = pd.read_csv("/home/debrupd1/XCOPA/hate_test_predictions/mBERT_mlm_edges.csv")    
    load_and_eval(df_testing= df,out_dir = local_dir)