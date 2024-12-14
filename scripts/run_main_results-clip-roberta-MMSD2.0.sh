#!/bin/bash

config_files_list=(
  'configs/mmsd2.0/clip-roberta/w-o-inter.yaml'
  'configs/mmsd2.0/clip-roberta/twoway.yaml'
  'configs/mmsd2.0/clip-roberta/v2t.yaml'
  'configs/mmsd2.0/clip-roberta/t2v.yaml'
)
for config_file in "${config_files_list[@]}"; do
    mmsd fit -c "$config_file" --data.max_length 256 --search-memo-size --run-test --result-save-path mmsd2-results/main-results-clip-roberta.csv
done
