import re
import os
import csv
import torch
import json
import pandas as pd
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


model_id = "meta-llama/Llama-2-13b-chat-hf"
pipeline = transformers.pipeline("text-generation", 
                                     model=model_id, 
                                    max_new_tokens=100,
                                     temperature=0.00001,
                                     model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

def load_prompt(prompt_file):
    with open(prompt_file) as f:
        prompt = f.read()
    return prompt

def get_llama_response(text):
    prompt = f"""Please label the sentiment of the given movie review text. The sentiment label should be "positive" or "negative". Answer only a single word for the sentiment label. Do not generate any extra text. \n\n Text: {text} \n Answer: """
    llama_output_original = pipeline(prompt)
    #print(llama_output_original)
    response = llama_output_original[0]['generated_text'].split('Answer:')[1].strip()
    if 'positive' in response.lower():
        pred = 1
    elif 'negative' in response.lower():
        pred = 0
    else:
        pred = -1
    return pred

if __name__ == "__main__":

    csv_file = pd.read_csv("validation_SST.csv")
    #original_prompt = load_prompt("prompt_SST.txt")
    json_file = [json.loads(lines) for lines in open("arms_MLM_xlmr_validation_SST.json", "r")]
    # df_preds = pd.DataFrame(columns = ['idx','pred'])

    # for index, row in tqdm(csv_file.iterrows()):
    #     text = row['Text'].strip()
    #     #prompt_output_original = original_prompt.replace("[[$$S$$]]", text)
    #     prompt_output_original = f"""Please label the sentiment of the given movie review text. The sentiment label should be "positive" or "negative". Answer only a single word for the sentiment label. Do not generate any extra text. \n\n Text: {text} \n Answer: """
    #     llama_output_original = pipeline(prompt_output_original)
    #     #print(llama_output_original)
    #     response = llama_output_original[0]['generated_text'].split('Answer: ')[1].strip()
    #     #print(response)
    #     if 'positive' in response.lower():
    #         pred = 1
    #     elif 'negative' in response.lower():
    #         pred = 0
    #     else:
    #         pred = -1
    #     df_preds = df_preds._append({'idx': index, 'pred':pred, 'response': response}, ignore_index=True)
    #     df_preds.to_csv('preds_SST_validation_2_13b.csv', index=False)

    for i in tqdm(range(len(json_file))):
        current_row = json_file[i]['meta']['samples']

        for j in range(len(current_row)):
            mlm_list = []

            sentence_index = current_row[j][0]
            original_sentence = csv_file.iloc[sentence_index]['Text'].strip()
            masked_sentences = current_row[j][2]
            mlm_preds = current_row[j][4]

            llama_output_original = get_llama_response(original_sentence)
            if(mlm_preds[0]==-1):
                llama_output_original = get_llama_response(original_sentence)
                mlm_preds[0] = llama_output_original
            for k, pred in enumerate(mlm_preds[1]):
                if(pred==-1):
                    new_sentence = masked_sentences[k].strip()
                    llama_output_mlm = get_llama_response(new_sentence)
                    mlm_preds[1][k]=llama_output_mlm

            
        with open("llama31_pred_sst_validation_errors.json", "a") as fl:
            json.dump(json_file[i], fl)
            fl.write('\n')
