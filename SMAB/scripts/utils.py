import re
import json
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from numpy.random import Generator, PCG64
from scipy.stats import norm, halfnorm, truncnorm
from itertools import combinations
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import scipy.stats as stats
import os
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
import evaluate


def read_stopwords(file_path):
    with open(file_path, 'r') as file:
        stopwords = file.read().splitlines()
    return set(stopwords)

def tokenize_sentence(sentence):
    words = sentence.split()
    cleaned_words = [word.rstrip(string.punctuation) for word in words]
    return cleaned_words
    

def remove_stopwords(tokens, stopwords):
    filtered_tokens = []
    for word in tokens:
        if word.lower() not in stopwords and not word.isdigit() and word not in string.punctuation and word.strip():
            filtered_tokens.append(word)

    return filtered_tokens


def reconstruct_sentence(filtered_tokens):
    return ' '.join(filtered_tokens)

def process_sentence(input_sentence, stopwords_file_path):
    stopwords = read_stopwords(stopwords_file_path)
    tokens = tokenize_sentence(input_sentence)
    filtered_tokens = remove_stopwords(tokens, stopwords)
    return filtered_tokens


def calculate_reward(pair):
    if pair[0] == pair[1]:
        return 0
    else:
        return 1

def average_reward(lst):
    
    if len(lst) <= 1:
        print("Single sample!:",lst)
        return None

    pairs = list(combinations(lst, 2))
    #print("List len:",len(lst))
    #print("Total pairs:",len(pairs))
    total_reward = sum(calculate_reward(pair) for pair in pairs)
    average = total_reward / len(pairs)
    return average

def calcprob_reward(lst,pred_l,prob_list):
    original_prob = prob_list[0]
    sample_probs_list = prob_list[1]
    
    if len(sample_probs_list) == 0:
        print("No samples")
        return None 
    else:
        flips = 0
        for k in range(len(sample_probs_list)):
            if lst[k] != pred_l:
                flips = flips + 1
            else:
                sample_prob = sample_probs_list[k]
                flips = flips + abs(sample_prob-original_prob)

        return flips/len(lst)
            


def flips_gold(lst,gold_l):
    
    if len(lst) == 0:
        print("No samples")
        return None 
    
    else:

        flips = 0
        for k in range(len(lst)):
            if lst[k] != gold_l:
                flips = flips + 1
        
        return flips/len(lst)

def get_flips(edge_mlm_pred_labels):
    flips = []
    flipped = []
    idxs = []
    #print(edge_mlm_pred_labels)
    for i, edge in enumerate(edge_mlm_pred_labels):
        pred_label = edge[0]
        mlm_labels = edge[1]
        for label in mlm_labels:
            if(label==pred_label):
                flips.append(0)
            else:
                flips.append(1)
                idxs.append(i)
                flipped.append(i)
    return flips, flipped, idxs
    
def calculate_perplexity(all_mlm_edges):
    
    perplexity = evaluate.load("perplexity", module_type="metric")
    results = perplexity.compute(model_id='gpt2',
                                 add_start_token=False,
                                 predictions=all_mlm_edges)
    perplexities = np.array(results["perplexities"])
    perplexities_inv = [(1/i) for i in perplexities]
    return np.array(perplexities_inv)/np.sum(perplexities_inv)



def plot_histogram_and_save(array,count,curr_iter,working_dir,setting,):
    print(working_dir)
    
    array_updated = []
    array = array.reshape(-1,1)
    for i in range(len(array)):
        if(count[i]>1):
            array_updated.append(array[i])
            
    array = np.array(array_updated)
            

    # Check if the input is a numpy array
    if not isinstance(array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    # Check if values are between 0 and 1
    if np.any((array < 0) | (array > 1)):
        raise ValueError("Values in the array must be between 0 and 1.")

    bin_edges = np.arange(0, 1.1, 0.1)

    # Use numpy.histogram to compute the histogram
    hist, _ = np.histogram(array, bins=bin_edges)    
    values = hist.tolist()
    #print(values)

    # Example data: list of values belonging to 10 categories
    categories = ['0.0', '0.1', '0.2', '0.3', '0.4',
                '0.5', '0.6', '0.7', '0.8', '0.9']


    # Define colors for each category
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'gray', 'cyan']

    # Create a bar plot
    sns.barplot(x=categories, y=values,hue=colors,legend=False)

    # Set labels and title
    plt.xlabel('Sensitivity category')
    plt.ylabel('No of Arms')
    plt.title('Global Sensitivity')

    # Save the plot
    plt.savefig(working_dir + setting + f'histplots/histogram_plot_gs_{curr_iter}.png')

    # Show the plot (optional)
    plt.clf()
    plt.close()