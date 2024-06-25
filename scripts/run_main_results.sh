#!/bin/bash

config_files_list=(
  'configs/interactive-ways/w-o-inter.yaml'
  'configs/interactive-ways/twoway.yaml'
  'configs/interactive-ways/v2t.yaml'
  'configs/interactive-ways/t2v.yaml'
)
for config_file in "${config_files_list[@]}"; do
    mmsd fit -c "$config_file" --search-memo-size --run-test --result-save-path ./main-results.csv
done
