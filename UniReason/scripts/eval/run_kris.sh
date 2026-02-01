# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

# export OPENAI_API_KEY=$openai_api_key

GPUS=8


# generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp_kris.py \
    --output_dir $output_path/bagel \
    --metadata_file ./eval/gen/kris/final_data.json \
    --max_latent_size 64 \
    --model-path $model_path \
    --think

