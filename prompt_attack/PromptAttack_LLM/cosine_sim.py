import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util

if __name__ == "__main__":
    '''
    P = "/home/sachin/prompting_attack/PromptAttack/without/sensitivity_word_level_new_index_6/"
    csvf = pd.read_csv(P + "attack_output.csv")
    #print(csvf.iloc[872]['Q'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    original_sent = []
    for i in range(872, len(csvf)):
        original_sent.append(csvf.iloc[i]['Q'].split("Only output the new sentence without anything else.")[1].split("  ->")[0].strip())
        
    generated_sent = csvf['V'][872: ].tolist()

    print(len(original_sent), len(generated_sent))

    assert len(original_sent) == len(generated_sent)
    
    embeddings1 = model.encode(original_sent, convert_to_tensor=True)
    embeddings2 = model.encode(generated_sent, convert_to_tensor=True)

    similarities = util.cos_sim(embeddings1, embeddings2).diagonal().tolist()

    overall_mean_similarity = np.mean(similarities)
    overall_variance_similarity = np.var(similarities)

    print(f"Overall Mean Similarity Score: {overall_mean_similarity:.4f}")
    print(f"Variance of Similarity Scores: {overall_variance_similarity:.4f}")

    plt.figure(figsize=(7, 6))
    sns.histplot(similarities, bins=20, kde=True, color='blue')
    plt.title('Histogram of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    
    histogram_path = P + "two_words_replacement.png"
    plt.savefig(histogram_path)
    print(f"Histogram saved to {histogram_path}")
    plt.close()
    '''
    
    # Define x-axis values
    x_values = [1, 2, 3, 4, 5, 6]
    
    # Define the two lists representing different metrics
    L1 = [0.416, 0.4846, 0.4200, 0.329, 0.277, 0.260]  # Example data for L1
    L2 = [0.877, 0.818, 0.769, 0.730, 0.705, 0.693]  # Example data for L2
    
    # Create a plot
    plt.figure(figsize=(5, 5))  # Set the size of the figure
    
    # Plot the first metric L1
    plt.plot(x_values, L1, label='Average ASR', marker='o', linestyle='-', color='blue')
    
    # Plot the second metric L2
    plt.plot(x_values, L2, label='Average Cosine Similarity', marker='x', linestyle='--', color='red')
    
    # Adding titles and labels
    plt.title('Average ASR vs Average  Cosine Similarity')
    plt.xlabel('Number of Words replaced.')
    plt.ylabel('ASR and Cosine Similarity')
    
    # Add a legend to differentiate the lists
    plt.legend()
    
    # Display the grid
    plt.grid(True)
    
    # Save the plot as an image file
    plt.savefig('metrics_comparison.png', bbox_inches='tight')  # Save as 'metrics_comparison.png' with high resolution
    


    
    





        