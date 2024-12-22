python mab_apply_MLM.py \
    --model_name "FacebookAI/xlm-roberta-large"\
    --folder_path "../data/dataset_sst/"\
    --csv_file "validation.csv"\
    --arms_file "arms_hashmap.json"\
    --output_file "arms_MLM_xlmr_validation.json"