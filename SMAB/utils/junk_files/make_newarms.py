import pandas as pd 
from tqdm import tqdm 
data_reward = [eval(lines) for lines in open("/home/debrupd1/XCOPA/hate_test_predictions/mBERT_mlm_samples.json", "r")]
print(data_reward[1])

df = pd.read_csv("/home/debrupd1/XCOPA/hate_test_predictions/mBERT_mlm_edges.csv")
df2 =  pd.read_csv("/home/debrupd1/XCOPA/hate_test_predictions/eng_dev_file_withprobs.csv")

new_arms = []

for i in tqdm(range(len(data_reward))):
    
    arm = data_reward[i]['meta']['word']
    data_reward[i]['index'] = i
    samples = data_reward[i]['meta']['samples']
    new_samples = []
    
    for edge in samples:
        ex_no = edge[0]
        new_words = edge[1]
        new_sents = edge[2]
        
        predictions_edge = []
        prob_edge = []

        for k in range(len(new_words)):
            new_w = new_words[k]
            match_df = df[(df['arm'] == arm) & (df['new_word'] == new_w) & (df['ex_no'] == ex_no)]

            pred_df = match_df['prediction']
            probs_df = match_df['probabilities']

            #print(pred_df)
            #print(probs_df)
            if len(match_df)>0:
              pred = pred_df.iloc[0]
              prob_ = (eval(probs_df.iloc[0]))[0]
              predictions_edge.append(pred)
              prob_edge.append(prob_)
              originalex_prob = eval(df2['probabilities'][ex_no])[0]
              originalex_pred = df2['prediction'][ex_no]
              l1 = [originalex_prob]
              l1.append(prob_edge)

              l2 = [originalex_pred]
              l2.append(predictions_edge)
        
              edge.append(l2)
              edge.append(l1)

            else:
              print(arm)
              print(new_w)
              print(ex_no)  
            
            #print(pred)
            #print(prob_)
           

            
            

        new_samples.append(edge)
    
    data_reward[i]['meta']['samples'] = new_samples
    new_arms.append(data_reward[i])



with open("arms_MLM_mBERT_withprobs.json", 'w') as file:
    
    # Dump each dictionary as a separate line in the file
    for dictionary in new_arms:
            #print(dictionary)
            #s = json.dumps(dictionary)
            s = str(dictionary)
            #json.dump(dictionary, file)
            #file.write(dictionary)
            
            file.write(s)
            file.write('\n')
    

        


