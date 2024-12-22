import torch
import numpy as np
import lime
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 1. Load the pretrained RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
class_names = ['neutral', 'negative', 'positive']

def predictor(texts):
    outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
    probas = F.softmax(outputs.logits).detach().numpy()
    print(probas)
    return probas

explainer = LimeTextExplainer(class_names=class_names)

str_to_predict = "You are a very good person"
exp = explainer.explain_instance(str_to_predict, predictor, num_features=20, num_samples=2000)
exp.save_to_file('lime.html')

# # Generate the visualization
# fig = exp.as_pyplot_figure()

# # Save the visualization to a file
# fig.savefig('lime_explanation.png')

# print(exp)