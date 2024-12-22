import pandas as pd
from collections import defaultdict
'''
def txt_to_csv(input_txt, output_csv):
    # Initialize a dictionary to store the data
    data = defaultdict(lambda: {'anger': 0, 'anticipation': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'negative': 0, 'positive': 0, 'sadness': 0, 'surprise': 0, 'trust': 0})
    
    # Read the input text file
    with open(input_txt, 'r') as file:
        for line in file:
            word, emotion, label = line.strip().split('\t')
            data[word][emotion] = float(label)
    
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(data, orient='index').reset_index()
    df.rename(columns={'index': 'word'}, inplace=True)
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)

# Specify the input and output files
input_txt = 'NRClexicon_files/NRC-Emotion-Intensity-Lexicon-v1.txt'  
output_csv = 'edgeweights_outputs/Feature_lexicon_based/NRCemotions_realvalues.csv'  
'''
# Convert the txt file to a csv file
#txt_to_csv(input_txt, output_csv)


'''
def extract_integer_and_add_column(input_csv, output_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)
    
    # Extract the integer from the 'probabilities' column
    df['LLMweight'] = df['probabilities'].str.extract(r'(\d+)').astype(int)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

# Specify the input and output file paths
input_csv = 'edgeweights_outputs/LLM_prompting_based/eng_dev_file_with_llmweights.csv'  
output_csv = 'edgeweights_outputs/LLM_prompting_based/eng_dev_file_with_llmweights_updated.csv'  

# Execute the function
extract_integer_and_add_column(input_csv, output_csv)
'''

df = pd.read_csv("edgeweights_outputs/LLM_prompting_based/eng_dev_file_with_llmweights_updated.csv")
index = 1761
print(df['prediction'][index])
print(df['LLMweight'][index])


