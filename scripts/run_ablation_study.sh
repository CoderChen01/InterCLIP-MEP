#!/bin/bash

config_files_list=(
  'configs/mmsd2.0/clip-base/w-o-inter.yaml'
  'configs/mmsd2.0/clip-base/t2v.yaml'
  'configs/mmsd2.0/clip-base/v2t.yaml'
  'configs/mmsd2.0/clip-base/twoway.yaml'
)
for config_file in "${config_files_list[@]}"; do
    # w/o proj.
    mmsd fit -c "$config_file" --model.use_sim_loss false --model.use_memo false  --run-test --result-save-path mmsd2-results/ablation_study.csv
    # w/o mep
    mmsd fit -c "$config_file"  --model.use_memo false --run-test --result-save-path mmsd2-results/ablation_study.csv
    # w/o lora
    mmsd fit -c "$config_file" --model.use_lora false --search-memo-size --run-test --result-save-path mmsd2-results/ablation_study.csv
done