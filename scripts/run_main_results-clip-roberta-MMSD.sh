#!/bin/bash

config_files_list=(
  'configs/mmsd/clip-roberta/w-o-inter.yaml'
  'configs/mmsd/clip-roberta/twoway.yaml'
  'configs/mmsd/clip-roberta/v2t.yaml'
  'configs/mmsd/clip-roberta/t2v.yaml'
)
for config_file in "${config_files_list[@]}"; do
    mmsd fit -c "$config_file" --data.max_length 256 --search-memo-size --run-test --result-save-path mmsd-results/main-results-clip-roberta.csv
done
