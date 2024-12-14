#!/bin/bash

config_files_list=(
  'configs/mmsd2.0/clip-base/w-o-inter.yaml'
  'configs/mmsd2.0/clip-base/twoway.yaml'
  'configs/mmsd2.0/clip-base/v2t.yaml'
  'configs/mmsd2.0/clip-base/t2v.yaml'
)
for config_file in "${config_files_list[@]}"; do
    mmsd fit -c "$config_file" --search-memo-size --run-test --result-save-path mmsd2-results/main-results.csv
done
