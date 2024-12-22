from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
import torch

# Importing required packages
from flask import Flask, request, jsonify

# Initiating a Flask application
app = Flask(__name__)

model_name = "meta-llama/Llama-2-7b-chat-hf"
pipe = pipeline("text-generation", 
                    model=model_name, 
                    max_new_tokens=256,
                    temperature=0.00001,
                    model_kwargs={"torch_dtype": torch.bfloat16}, 
                    device_map="auto")


def get_response(prompt):
    llama_output = pipe(prompt)
    response = llama_output[0]['generated_text'].strip()
    return response

# The endpoint of our flask app
@app.route('/predict', methods=['POST'])
def predict():
    # model_name = request.json['model_name']
    prompt = request.json['prompt']
    response = get_response(prompt)
    print(response)
    
    return jsonify({'response': response})

# Running the API
if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8081, debug=False)
