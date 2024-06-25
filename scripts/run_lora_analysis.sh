#!/bin/bash

# lora module
modules_list=(
  '["q_proj"]'
  '["k_proj"]'
  '["v_proj"]'
  '["out_proj"]'
  '["q_proj", "k_proj"]'
  '["q_proj", "v_proj"]'
  '["q_proj", "out_proj"]'
  '["k_proj", "v_proj"]'
  '["k_proj", "out_proj"]'
  '["v_proj", "out_proj"]'
  '["q_proj", "k_proj", "v_proj"]'
  '["q_proj", "k_proj", "out_proj"]'
  '["q_proj", "v_proj", "out_proj"]'
  '["k_proj", "v_proj", "out_proj"]'
  '["q_proj", "k_proj", "v_proj", "out_proj"]'
)
config_files_list=(
  'configs/interactive-ways/t2v.yaml'
  'configs/interactive-ways/v2t.yaml'
  'configs/interactive-ways/twoway.yaml'
  'configs/interactive-ways/w-o-inter.yaml'
)
for config_file in "${config_files_list[@]}"; do
  for modules in "${modules_list[@]}"; do
    mmsd fit -c "$config_file" --model.lora_modules "$modules" --search-memo-size --run-test --result-save-path ./lora_module_study.csv
  done
done
