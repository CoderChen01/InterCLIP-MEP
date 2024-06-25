#!/bin/bash

config_file="configs/best.yaml"

# lora rank
rank_list=(
  2
  4
  6
  8
  10
  12
)
for rank in "${rank_list[@]}"; do
  mmsd fit -c "$config_file" --model.lora_rank "$rank" --search-memo-size --run-test --result-save-path ./hyperparam_study.csv
done

# top-n
top_n_list=(
  2
  4
  6
  8
  10
  12
)
for top_n in "${top_n_list[@]}"; do
  mmsd fit -c "$config_file" --model.vision_cond_attn_mode "top-$top_n" --search-memo-size --run-test --result-save-path ./hyperparam_study.csv
done

# d_k
d_k_list=(
  128
  256
  384
  512
  640
  768
  896
  1024
  1152
  1280
)
for d_k in "${d_k_list[@]}"; do
  mmsd fit -c "$config_file" --model.embed_size "$d_k" --search-memo-size --run-test --result-save-path ./hyperparam_study.csv
done
