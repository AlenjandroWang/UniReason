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
    --master_port=12347 \
   ./eval/gen/gen_images_mp_wise.py \
    --output_dir $output_path/images \
    --metadata-file ./eval/gen/wise/final_data.json \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path $model_path \
    --think

