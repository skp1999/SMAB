
import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased',top_k = 10, device_map="cuda:0")
print("MLM mBERT pipeline initialized!")
data_reward = [eval(lines) for lines in open("./data/eng/new/xlmr_reward_withoutgold_mbert_eng.json", "r")]
#print(data_reward[1])

# Take input of the csv file 
df = pd.read_csv("./data/eng_dev_file_with_probs.csv")

for i in tqdm(range(len(data_reward))):
    
    singlearm_bool = True
    samples = data_reward[i]['meta']['samples']
    word = data_reward[i]['meta']['word']
    print("Word:",word)
    
    if ":" in word:
        singlearm_bool = False 
    
    # All edges of an arm 
    input_texts = [df['Text'][edge[0]] for edge in samples]
    print("Edges",input_texts)
    ex_nos = [edge[0] for edge in samples ]

    # Replace word by [MASK]
    if singlearm_bool is True:
        input_texts = [sent.replace(word, "[MASK]", 1) for sent in input_texts]
    
    #print(len(input_texts))

    with torch.no_grad():
           
           # Perform inference on the batch
           batch_results = unmasker(input_texts)
           #print(batch_results)
           
           if len(input_texts) == 1:
              batch_results = [batch_results]
           else:
              print("More than 1 edge!")   
              
           #print("MLM Completed for this Arm!")
           
           new_samples = []
           
           for j in range(len(batch_results)):
               
                edge_samples = []
                edge_sents = []
                edge_samplescores = []

                result_edge  = batch_results[j]
                #print(result_edge)
                
                for w in result_edge:
                    edge_samples.append(w['token_str'])
                    edge_sents.append(w['sequence'])
                    edge_samplescores.append(w['score'])

                
                new_samples.append([ex_nos[j],edge_samples,edge_sents,edge_samplescores])
    
    data_reward[i]['meta']['samples'] = new_samples 
    
    with open("mBERT_mlm_samples.json", 'a') as file:
        dictionary = data_reward[i]
        s = str(dictionary)
        file.write(s)
        file.write('\n')
        #print("Written to file for Arm!")
         
word = 'noble'
sentence =  'jewish journalist alanna schubach wants to bring multiculturalism to the noble japanese ending millennia of tradition and homogeneity.'       
         
def get_mbert_mlm_output(sentence, word):
    unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased',top_k = 10, device_map="cuda:0")
    print("MLM mBERT pipeline initialized!")
    input_text = sentence.replace(word, "[MASK]",1)

    with torch.no_grad():
        # Perform inference on the batch
        batch_results = unmasker(input_text)
        #print(batch_results)
        
        new_samples = []
        
        for j in range(len(batch_results)):
            
            edge_samples = []
            edge_sents = []
            edge_samplescores = []

            result_edge  = batch_results[j]
            #print(result_edge)
            
            for w in result_edge:
                edge_samples.append(w['token_str'])
                edge_sents.append(w['sequence'])
                edge_samplescores.append(w['score'])

            
            new_samples.append([ex_nos[j],edge_samples,edge_sents,edge_samplescores])
    
    return new_samples







                   


               
               



    
    
    
     



'''
        
from transformers import pipeline
import torch

# Define batch size
batch_size = 3

# Initialize the pipeline
unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased',top_k = 10)

# Input texts
input_texts = ["Hello I'm a [MASK] model.", "I like [MASK]."]

# Iterate over input_texts in batches
for i in range(0, len(input_texts), batch_size):
    batch_inputs = input_texts[i:i+batch_size]
    print(len(batch_inputs))
    with torch.no_grad():
        # Perform inference on the batch
        batch_results = unmasker(batch_inputs)
        print(len(batch_results))
        
'''