import re
import json
import random
import numpy as np
import pandas as pd
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
import torch
import random
import pickle
import argparse
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from utils import *
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Bandit:

    def __init__(self, K, N, hash_, gram_tuple,total_edges,
                 global_sensitivity = None,
                 setting="allcombinations/",
                 llama_true='no',
                 mbert_mlm=False,
                 prob_reward="no",
                 algorithm="ucb",
                 working_dir=None,
                 init_a=1,init_b=1,
                 LLM_edgeweight=None,
                 PPL_edgeweight=None,
                 NRC_edgeweight=None):
        ''' 
        K is the no. of arms/words in the outer bandit.
        N is the total no. of iterations for which the MAB setup will run.
        data is the data frame.
        hash is the dictionary that contains samples for every word.
        gram_tuple is the tuple of words/outer arms.
        highest_reward is the dictionary that contains the highest reward for every word.

        global_sensitivity is the list that contains the global_sensitivity of all the arms/words. Hence, its size is equal to K.
        count is the list that contains for every word/arm, the no. of times it has been pulled.
        total_regret is the total regret that we get while running the current MAB setup.
        reg_count_tuple is a dict that contains for every word, a tuple t = (current count of the word, global sensitivity at the current
        count)

        
        '''
        assert global_sensitivity is None or len(global_sensitivity) == K
        assert len(hash_) != 0
        assert len(gram_tuple) != 0 and len(gram_tuple) == len(hash_)
        
        self.K = K
        self.N = N
        self.total_edges = total_edges
        self.hash = hash_
        self.gram_tuple = gram_tuple
        self.count_singlesample = 0
        self.setting = setting
        self.llama_true = llama_true
        self.local_sens = {}
        self.mbert_mlm = mbert_mlm 
        self.prob_reward = prob_reward
        self.visited = {}  # Visited stats of every arm->edge
        self.less_edges = 0
        self.total_edges = 0
        self.algorithm = algorithm
        self.working_dir = working_dir
        #self.df_data = pd.read_csv("data/eng_dev_file_with_probs.csv")
        
        # Dataframe of LLM generated edge weights 
        #self.df_LLM_edgeweight = pd.read_csv("edgeweights_outputs/LLM_prompting_based/eng_dev_file_with_llmweights_updated.csv")
        
        # Dataframe of NRC Lexicon File
        #self.df1_NRC_edgeweight = pd.read_csv("edgeweights_outputs/Feature_lexicon_based/NRCemotions_binarylabels.csv")
        
        self.arm_weight_sum = [0 for _ in range(self.K)]
        
        self.LLM_edgeweight = LLM_edgeweight
        self.PPL_edgeweight = PPL_edgeweight
        self.NRC_edgeweight = NRC_edgeweight

        
        # Epsilon for Epsilon greedy approach in inner arms 
        self.epsilon = 0.9
         
        
        # If algorithm is Thompson Sampling 
        if self.algorithm == "thompson":
            #self._as = [init_a] * self.K
            #self._bs = [init_b] * self.K
            self._as = []
            self._bs = []
            
            for i in range(self.K):
                r = random.uniform(0,0.5)
                self._as.append(r)
                self._bs.append(1-r)
        
        for key_arm in self.hash:
            #print("No of edges in arm :",key_arm,len(self.hash[key_arm]))
            self.total_edges += len(self.hash[key_arm])
            if len(self.hash[key_arm])<3:
                self.less_edges +=1
            self.visited[key_arm] = [0]*len(self.hash[key_arm])
        
        print("No of arms having less than 3 edges",self.less_edges)
        
        if global_sensitivity is None:
            if self.algorithm == "ucb":
                self.global_sensitivity = [random.uniform(0, 1) for _ in range(self.K)]
                #self.global_sensitivity = list(truncnorm.rvs(a = 0, b = 0.1, size = self.K))
            if self.algorithm == "ucb_normalinit":   
                loc = 0.5
                scale = 0.05
                myclip_a = 0.45
                myclip_b = 0.55
                a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
                self.global_sensitivity = list(stats.truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=self.K))
            
            elif self.algorithm == "thompson":
                self.global_sensitivity = self.estimated_probas()  # Initialise the Global sensitivity values(All 0.5)
                #self.global_sensitivity = [random.uniform(0, 1) for _ in range(self.K)]
                
        else:
            self.global_sensitivity = global_sensitivity
        
        
        self.count = [1] * self.K    # For words/outer arms, index will be according to the gram_tuple not hash
        self.total_regret = 0
        self.regret_list = []
        self.reg_count_tuple = defaultdict(list)

    
    def estimated_probas(self):
        
        # Avg global sensitivity 
        return [self._as[i] / (self._as[i] + self._bs[i]) for i in range(self.K)]
    
    
    def select_global_arm(self, curr_iter):
        
        max_value = 0
        index_value = 0
        if curr_iter % 5000 == 0 or curr_iter>=(self.N)-1:
            plot_histogram_and_save(np.array(self.global_sensitivity),self.count, curr_iter,self.working_dir,self.setting)
        
        
        map_ndarray = []
        
        
        for k in range(len(self.count)):
            
            if self.visited[self.gram_tuple[k]] == []:
                map_ndarray.append(0) 
            elif self.count[k]-1 == 0 and len(self.visited[self.gram_tuple[k]])>0 and self.algorithm == "ucb":
                map_ndarray.append(100000000)
            else:      
                map_ndarray.append(1)

            
        
        # Print stats
        #print("Length count",len(self.count))
        print("Max visits of an arm", max(self.count)-1)
        print("Current iter",curr_iter)
                          


        # UCB Algorithm 
        if self.algorithm == "ucb" or self.algorithm =="ucb_normalinit":
            temp_gs_array = np.array(self.global_sensitivity) + np.sqrt((2 * np.log(curr_iter + 1))/ (np.array(self.count)))
            temp_gs_array = temp_gs_array * np.array(map_ndarray)
            max_value = np.max(temp_gs_array)
            index_value = np.argmax(temp_gs_array)
            word = self.gram_tuple[index_value]
            print("Index of selected arm:",index_value)
            print("UCB Value of selected arm",max_value)
            print("Sum of UCB values of all arms",sum(temp_gs_array.tolist()))
        
        
        elif self.algorithm == "thompson":
            
            # Take sample from Beta distribution
            samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.K) ] #* map_ndarray[x] for x in range(self.K) 
            index_value = max(range(self.K), key=lambda x: samples[x])
        return index_value

    def generate_reward(self, index, current_iter):
        
        # Get the reward value for the arm (by exploring an edge)

#             if self.LLM_edgeweight is None or self.LLM_edgeweight == "no":
#                 avg_local_value, _ = self.generate_local_sensitivity(index)

        if self.LLM_edgeweight == "yes" or self.NRC_edgeweight == "yes":
            avg_local_value, weight = self.generate_local_sensitivity(index)

        elif self.PPL_edgeweight == "yes":
            avg_local_value = self.generate_local_sensitivity(index)

        else:
            print('Not using any of LLM/PPL weights')
            avg_local_value, difference_reward = self.generate_local_sensitivity(index)


        # Print stats 
        print("No of edge visits of arm ",self.count[index]-1)
        print("Old GS Value" ,self.global_sensitivity[index])
        print("Avg local val",avg_local_value)
        print("Difference Reward",difference_reward)


        if self.algorithm == "ucb" or self.algorithm == "ucb_normalinit":
        # Update the total global sensitivity for the "index" word/arm by averaging associated local sensitivities.
            #self.global_sensitivity[index] = ((self.count[index]) * self.global_sensitivity[index] +avg_local_value)/(self.count[index] + 1)
            if self.LLM_edgeweight == "yes" or self.NRC_edgeweight == "yes":
                self.global_sensitivity[index] = self.global_sensitivity[index] * self.arm_weight_sum[index] + avg_local_value
                self.arm_weight_sum[index] += weight    # Increment weight sum 

                if (self.arm_weight_sum[index])!=0:
                    self.global_sensitivity[index] = self.global_sensitivity[index]/(self.arm_weight_sum[index]) # Rescale to have bw 0 and 1

            else:    
                self.global_sensitivity[index] = ((self.count[index]-1) * self.global_sensitivity[index] +
                                                  avg_local_value)/(self.count[index])

        elif self.algorithm == "thompson":
            self._as[index] += avg_local_value
            self._bs[index] += (1 - avg_local_value)
            self.global_sensitivity = self.estimated_probas()  # Update the Old Global sensitivity values





        self.count[index] += 1
        print("self.count updated for arm!")

        #print("global sensitivity of word with index {} is: {}".format(index, self.global_sensitivity[index]))
        self.reg_count_tuple[self.gram_tuple[index]].append((self.count[index], self.global_sensitivity[index]))



        #print("Avg local value is: ", avg_local_value)
        print("updated GS:",self.global_sensitivity[index])
            
        return difference_reward


    def get_sentences(self, curr_word):

        df = pd.read_csv("/home/debrupd1/XCOPA/hate_test_predictions/eng_dev_file.csv")
        examples = df['Text']
        cand_sent = [l for l in examples if curr_word in l]
        
        r = random.choice(cand_sent)
        return cand_sent
    
    
    def compute_NRC_wt(self,sent):
        
        #self.df1_NRC_edgeweight
        
        # Get weights using Vader Lexicon  (Valence  Aware  Dictionary  for SEntiment Reasoning) 
        #https://ojs.aaai.org/index.php/ICWSM/article/view/14550/14399.
        
        sid = SentimentIntensityAnalyzer()
        ss = sid.polarity_scores(sent)
        return (ss['neg'])  # Rule based method to get negative sentiment value
        
        
        '''
        # Remove stopwords from the sentence 
        sent = process_sentence(sent, "stopwords-en.txt")
        
        # Use the mode out of the words 
        mode_anger = None
        mode_negative = None
        mode_disgust = None
        
        for word in sent:
        '''
        
        
        
       
    def epsilon_greedy_NRC_edgeweight_reward(self,curr_word_edges) :
        
        
        all_mlm_edges, edge_mlm_pred_labels, all_mlm_NRCweights = [], [], []
        num_mlm_samples_list = []
        
        for edge in curr_word_edges:
            
            num_mlmsamples = 0
            
            example_no = edge[0]  # Get example no
            sent = self.df_data['Text'][int(example_no)]
            
            # Compute the edge weight using the NRC Lexicon
            edge_weight = self.compute_NRC_wt(sent)
            
            print("Edge wt:",edge_weight)
            
            edge_mlm_pred_labels.append(edge[4])
            
            if len(edge[2]) != len(edge[4][1]):
                print("Discrepency in json file")
                ch = input("Enter a key to continue!")
                
            
            for mlm_edge in edge[2]:
                num_mlmsamples += 1 
                
                all_mlm_edges.append(mlm_edge)
                all_mlm_NRCweights.append(edge_weight)
                
            num_mlm_samples_list.append(num_mlmsamples)   
            
        
        all_mlm_flips, all_mlm_flipped = get_flips(edge_mlm_pred_labels)

        print(f'All MLM edges: {all_mlm_edges}')
        print(f'All MLM LLM prompted edge weights: {all_mlm_NRCweights}')
        print(f'Edge and All MLM pred labels: {edge_mlm_pred_labels}')
        print(f'All flip labels: {all_mlm_flips}')

        print(len(all_mlm_edges), len(all_mlm_flips))
        
        if len(all_mlm_edges)!=len(all_mlm_flips):
               print("No of edges for picked arm:",len(edge_mlm_pred_labels))
               print("Number of MLM samples for each edge",num_mlm_samples_list)
               print("All MLM edges from flips",len(all_mlm_flips))
               print("All MLM edges from iteration",len(all_mlm_edges))
               ch = input("Enter a key to continue!")
        
        
        
        
        rand_no = random.uniform(0,1)
        print("Random no",rand_no)
        
        if rand_no < self.epsilon: # High epsilon means more biased towards max 
            
            # Find the maximum value
            max_value = max(all_mlm_flips)

            # Find the index of the maximum value
            max_index = all_mlm_flips.index(max_value)
            
            weight= all_mlm_NRCweights[max_index]
            flip = all_mlm_flips[max_index]

        else:    
            random_edge_index = random.randrange(start = 0, stop = len(all_mlm_flips))
            weight= all_mlm_NRCweights[random_edge_index]
            flip = all_mlm_flips[random_edge_index]

       
        reward = weight*flip 
        print("Reward",reward)
        print("Weight",weight)
        print("Flip",flip)
        return reward, weight
            
            
        

    
   
    
    
    def NRC_edgeweight_reward(self,curr_word_edges):
        
        all_mlm_edges, edge_mlm_pred_labels, all_mlm_NRCweights = [], [], []
        num_mlm_samples_list = []
        
        for edge in curr_word_edges:
            
            num_mlmsamples = 0
            
            example_no = edge[0]  # Get example no
            sent = self.df_data['Text'][int(example_no)]
            
            # Compute the edge weight using the NRC Lexicon
            edge_weight = self.compute_NRC_wt(sent)
            
            edge_mlm_pred_labels.append(edge[4])
            
            if len(edge[2]) != len(edge[4][1]):
                print("Discrepency in json file")
                ch = input("Enter a key to continue!")
                
            
            for mlm_edge in edge[2]:
                num_mlmsamples += 1 
                
                all_mlm_edges.append(mlm_edge)
                all_mlm_NRCweights.append(edge_weight)
                
            num_mlm_samples_list.append(num_mlmsamples)   
            
        
        all_mlm_flips, all_mlm_flipped = get_flips(edge_mlm_pred_labels)

        print(f'All MLM edges: {all_mlm_edges}')
        print(f'All MLM LLM prompted edge weights: {all_mlm_NRCweights}')
        print(f'Edge and All MLM pred labels: {edge_mlm_pred_labels}')
        print(f'All flip labels: {all_mlm_flips}')

        print(len(all_mlm_edges), len(all_mlm_flips))
        
        if len(all_mlm_edges)!=len(all_mlm_flips):
               print("No of edges for picked arm:",len(edge_mlm_pred_labels))
               print("Number of MLM samples for each edge",num_mlm_samples_list)
               print("All MLM edges from flips",len(all_mlm_flips))
               print("All MLM edges from iteration",len(all_mlm_edges))
               ch = input("Enter a key to continue!")
        
        
            
        random_edge_index = random.randrange(start = 0, stop = len(all_mlm_flips))
        weight= all_mlm_NRCweights[random_edge_index]
        flip = all_mlm_flips[random_edge_index]

        reward = weight*flip 
        return reward, weight     

        
    
    def LLM_edgeweight_reward(self,curr_word_edges):

        all_mlm_edges, edge_mlm_pred_labels, all_mlm_LLMweights = [], [], []
        num_mlm_samples_list = []
        
        for edge in curr_word_edges:
            # print(edge)
            num_mlmsamples = 0
            
            example_no = edge[0]  # Get example no
            edge_weight =  self.df_LLM_edgeweight['LLMweight'][example_no]
            
            edge_mlm_pred_labels.append(edge[4])
            
            if len(edge[2]) != len(edge[4][1]):
                print("Discrepency in json file")
                ch = input("Enter a key to continue!")
                
            
            for mlm_edge in edge[2]:
                num_mlmsamples += 1 
                
                all_mlm_edges.append(mlm_edge)
                all_mlm_LLMweights.append(edge_weight)
                
            num_mlm_samples_list.append(num_mlmsamples)    

        
        
        
        all_mlm_flips, all_mlm_flipped = get_flips(edge_mlm_pred_labels)

        print(f'All MLM edges: {all_mlm_edges}')
        print(f'All MLM LLM prompted edge weights: {all_mlm_LLMweights}')
        print(f'Edge and All MLM pred labels: {edge_mlm_pred_labels}')
        print(f'All flip labels: {all_mlm_flips}')

        print(len(all_mlm_edges), len(all_mlm_flips))
        
        if len(all_mlm_edges)!=len(all_mlm_flips):
               print("No of edges for picked arm:",len(edge_mlm_pred_labels))
               print("Number of MLM samples for each edge",num_mlm_samples_list)
               print("All MLM edges from flips",len(all_mlm_flips))
               print("All MLM edges from iteration",len(all_mlm_edges))
               ch = input("Enter a key to continue!")
        
        
            
        random_edge_index = random.randrange(start = 0, stop = len(all_mlm_flips))
        weight= all_mlm_LLMweights[random_edge_index]
        flip = all_mlm_flips[random_edge_index]

        reward = weight*flip 
        return reward, weight

    
    def epsilon_greedy_LLM_edgeweight_reward(self,curr_word_edges):
        
        all_mlm_edges, edge_mlm_pred_labels, all_mlm_LLMweights = [], [], []
        num_mlm_samples_list = []
        
        for edge in curr_word_edges:
            # print(edge)
            num_mlmsamples = 0
            
            example_no = edge[0]  # Get example no
            edge_weight =  self.df_LLM_edgeweight['LLMweight'][example_no]
            
            edge_mlm_pred_labels.append(edge[4])
            
            if len(edge[2]) != len(edge[4][1]):
                print("Discrepency in json file")
                ch = input("Enter a key to continue!")
                
            
            for mlm_edge in edge[2]:
                num_mlmsamples += 1 
                
                all_mlm_edges.append(mlm_edge)
                all_mlm_LLMweights.append(edge_weight)
                
            num_mlm_samples_list.append(num_mlmsamples)    

        
        
        
        all_mlm_flips, all_mlm_flipped = get_flips(edge_mlm_pred_labels)

        print(f'All MLM edges: {all_mlm_edges}')
        print(f'All MLM LLM prompted edge weights: {all_mlm_LLMweights}')
        print(f'Edge and All MLM pred labels: {edge_mlm_pred_labels}')
        print(f'All flip labels: {all_mlm_flips}')

        print(len(all_mlm_edges), len(all_mlm_flips))
        
        if len(all_mlm_edges)!=len(all_mlm_flips):
               print("No of edges for picked arm:",len(edge_mlm_pred_labels))
               print("Number of MLM samples for each edge",num_mlm_samples_list)
               print("All MLM edges from flips",len(all_mlm_flips))
               print("All MLM edges from iteration",len(all_mlm_edges))
               ch = input("Enter a key to continue!")
        
        
        
        rand_no = random.uniform(0,1)
        
        if rand_no < self.epsilon: # High epsilon means more biased towards max 
            
            # Find the maximum value
            max_value = max(all_mlm_flips)

            # Find the index of the maximum value
            max_index = all_mlm_flips.index(max_value)
            
            weight= all_mlm_LLMweights[max_index]
            flip = all_mlm_flips[max_index]

        else:    
            random_edge_index = random.randrange(start = 0, stop = len(all_mlm_flips))
            weight= all_mlm_LLMweights[random_edge_index]
            flip = all_mlm_flips[random_edge_index]

        
        reward = weight*flip 
        return reward, weight
        
        
    
    
    
    def epsilon_greedy_perplexity_reward(self, curr_word_edges):

        all_mlm_edges, edge_mlm_pred_labels, all_mlm_perplexities = [], [], []
    
        for edge in curr_word_edges:
            # print(edge)
            edge_mlm_pred_labels.append(edge[4])
            for mlm_edge in edge[2]:
                all_mlm_edges.append(mlm_edge)
            for mlm_perplexity in edge[5]:
                all_mlm_perplexities.append(mlm_perplexity)
                
        all_mlm_flips, all_mlm_flipped = get_flips(edge_mlm_pred_labels)

#         print(f'All MLM edges: {all_mlm_edges}')
#         print(f'All MLM perplexities: {all_mlm_perplexities}')
#         print(f'Edge and All MLM pred labels: {edge_mlm_pred_labels}')
#         print(f'All flip labels: {all_mlm_flips}')

        print(len(all_mlm_edges), len(all_mlm_flips))
        try:
            assert len(all_mlm_edges)==len(all_mlm_flips)
        except:
            return 0

        ### lM perplexity based reward
        #perplexities = calculate_perplexity(all_mlm_edges)
        
        random_edge_index = random.randrange(start = 0, stop = len(all_mlm_edges))
        ppl_max_index = np.argmax(all_mlm_perplexities)

        reward1 = all_mlm_perplexities[random_edge_index]
        flip1 = all_mlm_flips[random_edge_index]
        
        # reward2 = all_mlm_perplexities[ppl_max_index]
        # flip2 = all_mlm_flips[ppl_max_index]

        if(len(all_mlm_flipped)!=0):
            random_edge_index_flipped = random.randrange(start = 0, stop = len(all_mlm_flipped))
            reward2 = all_mlm_perplexities[random_edge_index_flipped]
            flip2 = 1
        else: 
            reward2 = 0
            flip2 = 0
        
        reward = (self.epsilon*reward1*flip1 + (1-self.epsilon)*reward2*flip2) / (self.epsilon*reward1+(1-self.epsilon)*reward2)
        print(f'Reward from epsilon-greedy perplexity: {reward}')
        
        return reward

    def epsilon_greedy_binary_reward(self, curr_word_edges):

        all_mlm_edges, edge_mlm_pred_labels, edge_pred_labels = [], [], []
        all_mlm_scores = []
        all_mlm_edges_cnt = 0
        
        for edge in curr_word_edges:
              all_mlm_edges_cnt += len(edge[1])
#             for mlm_edge in edge[2]:
#                 all_mlm_edges.append(mlm_edge)
#             for mlm_score in edge[3]:
#                 all_mlm_scores.append(mlm_score)
              edge_mlm_pred_labels.append(edge[4]) # 2 or 4
        
        #print(edge_mlm_pred_labels)
        all_mlm_flips, all_mlm_flipped, idxs = get_flips(edge_mlm_pred_labels)
        
         

        random_edge_index = random.randrange(start = 0, stop = min(all_mlm_edges_cnt, len(all_mlm_flips)))
        reward1 = all_mlm_flips[random_edge_index] #*all_mlm_scores[random_edge_index]
        highest_reward1 = max(all_mlm_flips)
        
        if(len(all_mlm_flipped)!=0):
            random_edge_index_flipped = random.randrange(start = 0, stop = len(all_mlm_flipped))
            reward2=1 #*all_mlm_scores[idxs[random_edge_index_flipped]]
            highest_reward2 = max(all_mlm_flipped)
        else:
            reward2=0
            highest_reward2 = 0

        if(reward1!=0 or reward2!=0):
            reward = (self.epsilon*reward1 + (1-self.epsilon)*reward2) / (self.epsilon*reward1+(1-self.epsilon)*reward2)
        else:
            reward = 0
            
        if(highest_reward1!=0 or highest_reward2!=0):
            high_reward = (self.epsilon*highest_reward1 + (1-self.epsilon)*highest_reward2) / (self.epsilon*highest_reward1+(1-self.epsilon)*highest_reward2)
        else:
            high_reward = 0
        
        print(f'Reward from epsilon-greedy binary: {reward}')
        print(f'Highest Reward from epsilon-greedy binary: {high_reward}')

        return reward, high_reward

    
    def generate_local_sensitivity(self, index):
        # select all sentences related to the "index" word from the hash and generate local sensitivity for that particular word
        # "index" word is at the "index" location in gram_tuple

        assert index < len(self.gram_tuple)
        average_local_sensitivity = 0
        curr_word = self.gram_tuple[index]
        curr_word_edges = self.hash[curr_word]
        print(f'No of edges for picked arm: {len(curr_word_edges)}')

        

        if len(curr_word_edges) != 0:
            
            if (self.NRC_edgeweight == "yes") and self.PPL_edgeweight == "yes" :
                weight = None
                average_local_sensitivity, weight = self.epsilon_greedy_NRC_edgeweight_reward(curr_word_edges)
                print(f'Reward for picked arm (LLM edge weighted): {average_local_sensitivity}')
                return average_local_sensitivity, weight
                
            
            elif (self.NRC_edgeweight == "yes"):
                weight = None
                average_local_sensitivity, weight = self.NRC_edgeweight_reward(curr_word_edges)
                print(f'Reward for picked arm (LLM edge weighted): {average_local_sensitivity}')
                return average_local_sensitivity,weight
                
            
            elif(self.LLM_edgeweight == "yes" and self.PPL_edgeweight == "no"):
                weight = None
                average_local_sensitivity, weight = self.LLM_edgeweight_reward(curr_word_edges)
                print(f'Reward for picked arm (LLM edge weighted): {average_local_sensitivity}')
                return average_local_sensitivity,weight
            
            
            elif (self.LLM_edgeweight == "yes" and self.PPL_edgeweight == "yes"):
                weight = None
                average_local_sensitivity, weight = self.epsilon_greedy_LLM_edgeweight_reward(curr_word_edges)
                print(f'Reward for picked arm (LLM edge weighted): {average_local_sensitivity}')
                return average_local_sensitivity,weight
                
            
            elif(self.PPL_edgeweight == "yes") :
                average_local_sensitivity = self.epsilon_greedy_perplexity_reward(curr_word_edges)
                print(f'Average Local Sensitivity: {average_local_sensitivity}')
                return average_local_sensitivity
            
            else:
                average_local_sensitivity, high_reward = self.epsilon_greedy_binary_reward(curr_word_edges)
                print(f'Average Local Sensitivity: {average_local_sensitivity}')
                
                difference_reward = high_reward - average_local_sensitivity
                print(f'Difference reward : {difference_reward}')
                
                return average_local_sensitivity, difference_reward
        else:
            return 0, 1
        
    def update_regret(self, index, d_reward):

        self.total_regret += (d_reward * self.global_sensitivity[index])
        print(f'Total regret: {self.total_regret}')
        self.regret_list.append(self.total_regret)   
    
    def run(self):
        executed = 0
        for j in tqdm(range(self.N)):
            print("-"*60)
        
            i_val = self.select_global_arm(j)
            curr_word = self.gram_tuple[i_val]
            
            #self.generate_reward(i_val, j)
            d_reward = self.generate_reward(i_val, j)
            self.update_regret(i_val, d_reward)
            
#             v_i = self.visited[curr_word]
#             flag=0
#             for v in v_i: 
#                 if v<1:
#                     print("At least one edge not Visited present!")
#                     print("Edge visited, List of visit:",v,v_i)
#                     self.generate_reward(i_val, executed)
#                     executed+=1
#                     flag=1
#                     break
            
#             if flag==0:
#                 print("Index of culprit arm",i_val)
#                 print("No of times culprit arm visited",self.count[i_val]-1)
#                 print("No of edges of culprit arm",len(self.visited[self.gram_tuple[i_val]]))
#                 if self.visited[self.gram_tuple[i_val]] == [1]*len(self.visited[self.gram_tuple[i_val]]):
#                     print("All edges visited of culprit arm!")
#                 print("Visited Edges list of culprit arm",self.visited[self.gram_tuple[i_val]])
            
#             print("No of times executed:",executed)    
            print("Total edges in bandit:",self.total_edges)
            print("-"*60)
        plot_histogram_and_save(np.array(self.global_sensitivity),self.count, executed,self.working_dir,self.setting)
            #print("*********************************************************************************")
        
        
           
            
