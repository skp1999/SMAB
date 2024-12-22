from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
import torch

# Importing required packages
from flask import Flask, request, jsonify

# Initiating a Flask application
app = Flask(__name__)


def define_bnb_config():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    return bnb_config

def load_model(model_name):
    bnb_config = define_bnb_config()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
    
    return tokenizer, model

model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer, model = load_model(model_name)

def get_response(prompt, max_new_tokens, do_sample, temperature):

    conversation = [
                {"role": "user", "content": f"{prompt}"}
            ]
    
    terminators = [
        tokenizer.eos_token_id,  # End-of-sentence token
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # Custom end-of-conversation token
        ]

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    prompt_padded_len = len(input_ids[0])
  
    # Generate text with specified parameters
    generate_kwargs = {
        "input_ids": input_ids,
        "max_length": max_new_tokens + input_ids.shape[1],  # Adjust for total length
        "do_sample": do_sample,  # Use sampling for non-zero temperature (randomness)
        "temperature": temperature,
        "eos_token_id": terminators,  # Specify tokens to stop generation
    }

    # Generate output tokens and decode them to text
    output = model.generate(**generate_kwargs)
    output_tokens = [gt[prompt_padded_len:] for gt in output]
    
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    
    return output_text

# The endpoint of our flask app
@app.route('/predict', methods=['POST'])
def predict():
    # model_name = request.json['model_name']
    prompt = request.json['prompt']
    max_new_tokens = request.json['max_new_tokens']
    do_sample = request.json['do_sample']
    temperature = request.json['temperature']

    response = get_response(prompt, 
                            max_new_tokens=max_new_tokens,
                            do_sample=do_sample,
                            temperature=temperature)
    
    return jsonify({'response': response})

# Running the API
if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8081, debug=False)
