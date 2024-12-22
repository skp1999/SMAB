
import pandas as pd
from transformers import pipeline
from transformers import BertTokenizer, BertModel,BertForMaskedLM
import torch
from tqdm import tqdm
import numpy as np
import random
import os
import pke

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)


class bert_predict_mlm():

    def __init__(self):
        
        self.device = 'cuda:0'
        self.unmasker = pipeline('fill-mask', 
                            model='bert-base-multilingual-cased',
                            top_k = 10, 
                            device_map=self.device)
        print("mBERT MLM pipeline initialized!")

        mlm_multimask_model_name = 'bert-large-uncased'
        self.model = BertForMaskedLM.from_pretrained(mlm_multimask_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(mlm_multimask_model_name)    
         
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
    
    def get_mbert_mlm_multimask_output(self, sentence):

        mlm_sentences = list()
        for i in range(10):
            sentence = f'[CLS] {sentence} [SEP]'
            tokenized_text = self.tokenizer.tokenize(sentence)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            outputs = self.model(tokens_tensor)
            predictions = outputs[0]
            predicted_index = [torch.argmax(predictions[0, i]).item() for i in range(0,predictions.shape[1])]
            mlm_output = self.tokenizer.decode(predicted_index)
            mlm_output = mlm_output[1:-1]
            mlm_sentences.append(mlm_output)
        
        return mlm_sentences

def extract_keyphrases(sample):

    # initialize a TopicRank keyphrase extraction model
    extractor = pke.unsupervised.TopicRank()

    extractor.load_document(input=sample, language='en')
    extractor.candidate_selection()
    extractor.candidate_weighting()
    # let's have a look at the topics

    keyphrases = extractor.get_n_best(stemming=False)
    return keyphrases

arms = ["ostentatious", "portraying", "aristocracy", "downfall", "charming", "elegance", "profligate"]
mlm_obj = bert_predict_mlm()
samples = ["In portraying the ostentatious and self-indulgent aristocracy who bring about their own downfall, the film highlights the charming elegance of a bygone era. It is no wonder that their financial troubles catch up with them, given their profligate ways and lack of concern for anyone but themselves."]
for sample in samples:
    keyphrases = extract_keyphrases(sample)
    for i, keyphrase in enumerate(keyphrases):
        flag=False
        words = keyphrase[0].split()
        for word in words:
            if(flag==True):
            if word in arms:
                sample = sample.replace(word, "[MASK]",1)
        mlm_sentences = mlm_obj.get_mbert_mlm_multimask_output(sample1)
        print('#################')
        print(sample1)
        print()
        print(mlm_sentences)