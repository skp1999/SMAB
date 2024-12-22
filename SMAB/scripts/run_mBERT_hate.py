import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, TensorDataset, DataLoader
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

class mBERT_predict_hate():

    def __init__(self):
        
        self.device = 'cuda:0'

        print("Loading Tokenizer")
        self.loaded_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        print("Loading Model")    
        self.loaded_model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", 
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
        for batch_idx, (Input_ids, mask_ids) in tqdm(enumerate(tuple(zip(input_ids, attention_masks)))):
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
    