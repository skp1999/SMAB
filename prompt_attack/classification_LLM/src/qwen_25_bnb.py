from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
import pandas as pd
import json
from tqdm import tqdm


name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)

# text = "any text "    # prompt goes here

def get_llama_response(text):
    prompt = f"""Please label the sentiment of the given movie review text. The sentiment label should be "positive" or "negative". Answer only a single word for the sentiment label. Do not generate any extra text. \n\n Text: {text} \n Answer: """
    #llama_output_original = pipeline(prompt)

    llama_output_original = generation_pipe(
        prompt,
        max_length=128,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.00001,
    )
    
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
    json_file = [json.loads(lines) for lines in open("arms_MLM_xlmr_validation_SST.json", "r")]

    for i in tqdm(range(len(json_file))):
        current_row = json_file[i]['meta']['samples']

        for j in range(len(current_row)):
            mlm_list = []

            sentence_index = current_row[j][0]
            original_sentence = csv_file.iloc[sentence_index]['Text'].strip()
            masked_sentences = current_row[j][2]

            llama_output_original = get_llama_response(original_sentence)

            for k in range(len(masked_sentences)):
                New_sentence = masked_sentences[k].strip()
                llama_output_mlm = get_llama_response(New_sentence)
                mlm_list.append(llama_output_mlm)

            current_row[j].append([llama_output_original, mlm_list])

            
        with open("qwen_25_pred_sst_validation.json", "a") as fl:
            json.dump(json_file[i], fl)
            fl.write('\n')