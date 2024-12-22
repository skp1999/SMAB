import re
import os
import csv
import json
import time
import openai
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

def load_prompt(prompt_file):
    with open(prompt_file) as f:
        prompt = f.read()
    return prompt

def generate_output(Prompt):
    response = openai.ChatCompletion.create(
        engine = "gpt-4-turbo",
        messages = [{"role": "user", "content": Prompt}],
        temperature = 0.0,
        max_tokens = 10,
        top_p = 1.0,
        frequency_penalty = 0,
        presence_penalty = 0,
        stop = '------'
    )
    generated_text = response['choices'][0]['message']['content'].strip()
    return generated_text

def get_gpt_output(Final_prompt, index):
    try:
        gpt_output = int(generate_output(Final_prompt))
    except Exception as e:
        gpt_output = -1
        print("The exception {} occured at index {}.".format(e, index))
    return gpt_output
    

if __name__ == "__main__":

    engine = "gpt-4-turbo"
    openai.api_type = "azure"
    openai.api_version = "2024-02-15-preview"
    openai.api_key = ''
    openai.api_base = ''
    
    df = pd.read_csv("results/output_csvs_7B/attack_3_output_final_preds.csv")
    df = df.fillna('')
    df['gpt4'] = None
    
    original_prompt = load_prompt("prompt_SST.txt")

    for index, row in tqdm(df.iterrows()):
        text = str(row['paraphrase_text'])
        final_prompt = original_prompt.replace("[[$$S$$]]", text)
        response = get_gpt_output(final_prompt, index)
        df.loc[index, 'gpt4'] = response
        df.to_csv('results/output_csvs_7B/attack_3_output_final_gpt4_preds.csv', index=False)
    
    # Gpt4_prediction_list = []
    # for i in tqdm(range(len(csv_file))):#
    #     New_sentence = csv_file.iloc[i]['paraphrase_text']
    #     Final_prompt = original_prompt.replace("[[$$S$$]]", New_sentence)
    #     gpt4_output = get_gpt_output(final_prompt, engine, i)
    #     Gpt4_prediction_list.append(gpt4_output)
    #     time.sleep(0.15)

    # csv_file['gpt4_preds'] = Gpt4_prediction_list
    # csv_file.to_csv("results/output_csvs/attack_10_output_final_gpt4_preds.csv", index = False)

    # for index, row in tqdm(df.iterrows()):
    #         text = row['paraphrase_text']
    #         response = get_llama_response(text)
    #         df.loc[index, 'pred'] = response
    #         df.to_csv('results/output_csvs/attack_8_output_final_preds.csv', index=False)

        

                



    

    

    