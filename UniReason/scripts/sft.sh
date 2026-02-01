#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
NUM_NODES=8
NPROC_PER_NODE=8
MODEL_PATH=your_model_path/BAGEL-7B-MoT
PARM_PATH=your_model_path/results/stage_1_align/0030000

# replace the variables with your own
torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=$NPROC_PER_NODE \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example_sft.yaml \
  --model_path $MODEL_PATH \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $PARM_PATH \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --ema 0.995 \
  --ce_weight 2.0 \
  --lr_scheduler cosine \
  --min_lr 1e-6 \
  --timestep_shift 4 \
  --num_worker 4 \
  --max_num_tokens 50000 \
  --cpu_offload True \
  --max_num_tokens_per_sample 50000 \
  --prefer_buffer_before 50000 \
  --sharding_strategy="FULL_SHARD" \
  --save_every 5000 \
  --warmup_steps 1000 \
  --total_steps 10000 \
  --results_dir your_model_path/results/ \
  --checkpoint_dir your_model_path/results/stage_2_reason/ > run_sft.out 2> run_sft.err  \

# --cpu_offload True \