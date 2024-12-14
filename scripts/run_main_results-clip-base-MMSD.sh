#!/bin/bash

config_files_list=(
  'configs/mmsd/clip-base/w-o-inter.yaml'
  'configs/mmsd/clip-base/twoway.yaml'
  'configs/mmsd/clip-base/v2t.yaml'
  'configs/mmsd/clip-base/t2v.yaml'
)
for config_file in "${config_files_list[@]}"; do
    mmsd fit -c "$config_file" --search-memo-size --run-test --result-save-path mmsd-results/main-results.csv
done
