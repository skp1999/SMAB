
import pandas as pd
data_reward = [eval(lines) for lines in open("/home/debrupd1/XCOPA/hate_test_predictions/arms_MLM_mBERT_withprobs.json", "r")]
print(data_reward[1])

df = pd.read_csv("/home/debrupd1/XCOPA/hate_test_predictions/eng_dev_file_withprobs.csv")
examples_probs = df['probabilities']

new_arms = []
for i in range(len(data_reward)):
    
    '''
    if len(data_reward[i]['meta']['samples']) >= 3:
        new_arms.append(data_reward[i])
    '''
    samples = data_reward[i]['meta']['samples']
    new_samples = []
    
    for edge in samples:
        ex_no = edge[0]
        prob_list = eval(examples_probs[ex_no])
        pred_ = df['prediction'][ex_no]
        gold_ = df['label'][ex_no]
        print(prob_list)
        if  prob_list[0]>= 0.45 and prob_list[0]<= 0.55 and pred_!=gold_ :
            new_samples.append(edge)
    
    if len(new_samples) == 0:
         continue
    
    data_reward[i]['meta']['samples'] = new_samples
    new_arms.append(data_reward[i])


             

with open("mbertmlm_arms_lowconfidence_pred_ne_gold.json", 'w') as file:
    
    # Dump each dictionary as a separate line in the file
    for dictionary in new_arms:
            #print(dictionary)
            #s = json.dumps(dictionary)
            s = str(dictionary)
            #json.dump(dictionary, file)
            #file.write(dictionary)
            
            file.write(s)
            file.write('\n')



#python bandit_run.py -r "/home/debrupd1/XCOPA/hate_test_predictions/mbertmlm_arms_pred_ne_gold.json" -s "mbertmlm_predicted_all_visit1_pred_neq_gold_20000/" -m yes
#python bandit_run.py -r "/home/debrupd1/XCOPA/hate_test_predictions/mbertmlm_arms_pred_eq_gold.json" -s "mbertmlm_predicted_all_visit1_pred_eq_gold_20000/" -m yes