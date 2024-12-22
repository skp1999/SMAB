# Importing required packages
import requests
import json
import argparse
import pickle
from tqdm import tqdm
import pandas as pd

# Initialize parser
parser = argparse.ArgumentParser()

# Adding arguments
#parser.add_argument("-m", "--model_name", help = "LLM name", default="gpt-3.5-turbo")
parser.add_argument("-ip", "--input_file", help = "path to data", default="results/output_csvs_13B/attack_9_output_final.csv")
parser.add_argument("-op", "--output_file", help = "path to output file", default="results/output_csvs_13B/attack_9_output_final.csv")


# Read arguments from command line
args = parser.parse_args()

INPUT_URL = "10.5.30.82"
INPUT_PORT = 8081

def run_client(payload):
    
    # Defining content type for our payload
    headers = {'Content-type': 'application/json'}

    # Sending a post request to the server (API)
    response = requests.post(url=f"http://{INPUT_URL}:{INPUT_PORT}/predict", 
                            data=json.dumps(payload), 
                            headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.json()}"

def get_llama_response(text):
    prompt = f"""Please label the sentiment of the given movie review text. The sentiment label should be "positive" or "negative". Answer only a single word for the sentiment label. Do not leave answer as empty. Do not generate any extra text. \n\n Text: {text} \n Answer: """
    payload = {
        "prompt": prompt,
    }
    llama_output_original = run_client(payload)['response']
    response = llama_output_original.split('Answer:')[1].strip()
    if 'positive' in response.lower():
        pred = 1
    elif 'negative' in response.lower():
        pred = 0
    else:
        pred = -1
    return pred

    
if __name__ == "__main__":

    df = pd.read_csv(args.input_file)
    df['original_pred'] = None # paraphrase_pred, original_pred

    for index, row in tqdm(df.iterrows()):
        text = row['Text'] # Text, paraphrase_text
        response = get_llama_response(text)
        df.loc[index, 'original_pred'] = response
        df.to_csv(args.output_file, index=False)
        
    


