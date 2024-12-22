# SMAB: MAB based word Sensitivity Estimation Framework and its Applications in Adversarial Text Generation

This is the official code repository for the paper SMAB: MAB based word Sensitivity Estimation Framework and its Applications in Adversarial Text Generation (link when accepted), accepted at [conference name].

*Saurabh Kumar Pandey, Sachin Vashishtha, Debrup Das, Somak Aditya, Monojit Choudhury* 

## Abstract
To understand the complexity of sequence classification tasks, Hahn et al. (2021) proposed sensitivity as the number of disjoint subsets of the input sequence that can each be individually changed to change the output. Though effective, calculating sensitivity at scale using this framework is costly because of exponential time complexity. Therefore, we introduce a Sensitivity-based Multi-Armed Bandit framework (SMAB), which provides a scalable approach for calculating word-level local (sentence-level) and global (aggregated) sensitivities concerning an underlying text classifier for any dataset. We establish the effectiveness of our approach through various applications. We perform a case study on CHECKLIST generated sentiment analysis dataset where we show that our algorithm indeed captures intuitively high and low-sensitive words. Through experiments on multiple tasks and languages, we show that sensitivity can serve as a proxy for accuracy in the absence of gold data. Lastly, we show that guiding perturbation prompts using sensitivity values in adversarial example generation improves attack success rate by 13.61%, whereas using sensitivity as an additional reward in adversarial paraphrase generation gives a 12.00% improvement over SOTA approaches. 

[Note] Code release is in progress. Stay tuned !!!

## Environment Setup

## Steps to run PromptAttack using LLMs

1. Run SMAB/utils/mab_get_arms.py to get all the arms in a dataset.
2. Run SMAB/utils/mab_apply_mlm.sh to generate the inner arms (MLM sentences) using a MLM model and suitable params.
3. Use an inference file from the folder prompt_attack/classification_LLM/src/ with suitable data paths (.json generated in 2) to get the LLM predictions on all the arms.
4. Run SMAB/scripts/bandit_run.sh with suitable arguments to get the output estimated sensitivity values from SMAB framework. 
4. Host the server (any open source or closed source).
5. Use the server endpoint and in prompt_attack/PromptAttack_LLM/run.sh to attack various LLMs and get the .db files saved.
6. Run LLM (llama/gpt) using llama_client.py ot llama.sh on the generated outputs to get the predictions for calculating ASR.
7. Use results.ipynb to calculate the ASR and other metrics.