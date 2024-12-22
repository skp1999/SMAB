# python -W ignore bandit_run.py \
#   --setting thompson_binary_mhate_english_mbert_200k/ \
#   --algorithm thompson \
#   --num_iters 200000 \
#   --reward_file data/dataset_mhate/english/new/xlmr_reward_withoutgold_mbert_eng_updated.json \
#   --data_file data/dataset_mhate/english/new/valid.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"
  
python -W ignore bandit_run_regret.py \
  --setting thompson_binary_sst_english_gpt_regret_200k/ \
  --algorithm thompson \
  --num_iters 200000 \
  --reward_file data/dataset_sst/arms_MLM_xlmr_validation_preds_gpt.json \
  --data_file data/dataset_sst/validation.csv \
  --mbert_mlm "yes" \
  --probreward "yes" \
  --LLM_edgeweight "no" \
  --PPL_edgeweight "no"
  
python -W ignore bandit_run_regret.py \
  --setting thompson_binary_sst_english_llama2_7B_regret_200k/ \
  --algorithm thompson \
  --num_iters 200000 \
  --reward_file data/dataset_sst/arms_MLM_xlmr_validation_preds_llama2_7B.json \
  --data_file data/dataset_sst/validation.csv \
  --mbert_mlm "yes" \
  --probreward "yes" \
  --LLM_edgeweight "no" \
  --PPL_edgeweight "no"
  
# python -W ignore bandit_run_regret.py \
#   --setting thompson_binary_cl_regret_50k/ \
#   --algorithm thompson \
#   --num_iters 50000 \
#   --reward_file data/checklist/checklist_all_types_1000/arms_MLM_roberta_base_uncased_validation_preds.json \
#   --data_file data/checklist/checklist_all_types_1000/checklist_all_types_1000.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"
  
# python -W ignore bandit_run_regret.py \
#   --setting ucb_binary_cl_regret_50k/ \
#   --algorithm ucb \
#   --num_iters 50000 \
#   --reward_file data/checklist/checklist_all_types_1000/arms_MLM_roberta_base_uncased_validation_preds.json \
#   --data_file data/checklist/checklist_all_types_1000/checklist_all_types_1000.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"


# python -W ignore bandit_run.py \
#   --setting ucb_binary_hate_english_mbert_bengali_200k/ \
#   --algorithm ucb \
#   --num_iters 200000 \
#   --reward_file data/dataset_xnli/XNLI/mab_data/english/test/deberta_rewards_gold.json \
#   --data_file data/dataset_multilingual/bengali/new/valid.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"
  
# python -W ignore bandit_run.py \
#   --setting ucb_binary_hate_english_mbert_english_200k/ \
#   --algorithm ucb \
#   --num_iters 200000 \
#   --reward_file data/dataset_multilingual/english/new/xlmr_reward_withoutgold_mbert_eng_updated.json \
#   --data_file data/dataset_multilingual/english/new/valid.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"

# python -W ignore bandit_run.py \
#   --setting ucb_binary_hate_english_mbert_french_200k/ \
#   --algorithm ucb \
#   --num_iters 200000 \
#   --reward_file data/dataset_multilingual/french/new/xlmr_reward_withoutgold_mbert_eng_updated.json \
#   --data_file data/dataset_multilingual/french/new/valid.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"
  
# python -W ignore bandit_run.py \
#   --setting ucb_binary_hate_english_mbert_german_200k/ \
#   --algorithm ucb \
#   --num_iters 200000 \
#   --reward_file data/dataset_multilingual/german/new/xlmr_reward_withoutgold_mbert_eng_updated.json \
#   --data_file data/dataset_multilingual/german/new/valid.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"

# python -W ignore bandit_run.py \
#   --setting ucb_binary_hate_english_mbert_greek_200k/ \
#   --algorithm ucb \
#   --num_iters 200000 \
#   --reward_file data/dataset_multilingual/greek/new/xlmr_reward_withoutgold_mbert_eng_updated.json \
#   --data_file data/dataset_multilingual/greek/new/valid.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"
  
 
   
