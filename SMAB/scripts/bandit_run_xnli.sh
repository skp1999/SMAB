# python -W ignore bandit_run.py \
#   --setting thompson_0.5_0.9_binary_xnli_english_200k/ \
#   --algorithm thompson \
#   --num_iters 200000 \
#   --reward_file data/dataset_xnli/XNLI/mab_data/english/test/deberta_rewards_gold_updated.json \
#   --data_file data/dataset_xnli/english/english_test.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"

# python -W ignore bandit_run.py \
#   --setting thompson_0.5_0.9_binary_xnli_french_200k/ \
#   --algorithm thompson \
#   --num_iters 200000 \
#   --reward_file data/dataset_xnli/XNLI/mab_data/french/test/deberta_rewards_gold_updated.json \
#   --data_file data/dataset_xnli/french/french_test.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"
  
# python -W ignore bandit_run.py \
#   --setting thompson_0.5_0.9_binary_xnli_german_200k/ \
#   --algorithm thompson \
#   --num_iters 200000 \
#   --reward_file data/dataset_xnli/XNLI/mab_data/german/test/deberta_rewards_gold_updated.json \
#   --data_file data/dataset_xnli/german/german_test.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"
  
# python -W ignore bandit_run.py \
#   --setting thompson_0.5_0.9_binary_xnli_greek_200k/ \
#   --algorithm thompson \
#   --num_iters 200000 \
#   --reward_file data/dataset_xnli/XNLI/mab_data/greek/test/deberta_rewards_gold_updated.json \
#   --data_file data/dataset_xnli/greek/greek_test.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"
  
python -W ignore bandit_run.py \
  --setting thompson_0.5_0.9_binary_xnli_hindi_200k/ \
  --algorithm thompson \
  --num_iters 200000 \
  --reward_file data/dataset_xnli/XNLI/mab_data/hindi/test/deberta_rewards_gold_updated.json \
  --data_file data/dataset_xnli/hindi/hindi_test.csv \
  --mbert_mlm "yes" \
  --probreward "yes" \
  --LLM_edgeweight "no" \
  --PPL_edgeweight "no"
  
# python -W ignore bandit_run.py \
#   --setting thompson_0.5_0.9_binary_xnli_spanish_200k/ \
#   --algorithm thompson \
#   --num_iters 200000 \
#   --reward_file data/dataset_xnli/XNLI/mab_data/spanish/test/deberta_rewards_gold_updated.json \
#   --data_file data/dataset_xnli/spanish/spanish_test.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"
  
# python -W ignore bandit_run.py \
#   --setting thompson_0.5_0.9_binary_xnli_swahili_200k/ \
#   --algorithm thompson \
#   --num_iters 200000 \
#   --reward_file data/dataset_xnli/XNLI/mab_data/swahili/test/deberta_rewards_gold_updated.json \
#   --data_file data/dataset_xnli/swahili/swahili_test.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"
  
# python -W ignore bandit_run.py \
#   --setting thompson_0.5_0.9_binary_xnli_turkish_200k/ \
#   --algorithm thompson \
#   --num_iters 200000 \
#   --reward_file data/dataset_xnli/XNLI/mab_data/turkish/test/deberta_rewards_gold_updated.json \
#   --data_file data/dataset_xnli/turkish/turkish_test.csv \
#   --mbert_mlm "yes" \
#   --probreward "yes" \
#   --LLM_edgeweight "no" \
#   --PPL_edgeweight "no"
