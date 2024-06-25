#!/bin/bash

config_files_list=(
  'configs/interactive-ways/w-o-inter.yaml'
  'configs/interactive-ways/t2v.yaml'
  'configs/interactive-ways/v2t.yaml'
  'configs/interactive-ways/twoway.yaml'
)
for config_file in "${config_files_list[@]}"; do
    # w/o proj.
    mmsd fit -c "$config_file" --model.use_sim_loss false --model.use_memo false  --run-test --result-save-path ./ablation_study.csv
    # w/o mep
    mmsd fit -c "$config_file"  --model.use_memo false --run-test --result-save-path ./ablation_study.csv
    # w/o lora
    mmsd fit -c "$config_file" --model.use_lora false --search-memo-size --run-test --result-save-path ./ablation_study.csv
done