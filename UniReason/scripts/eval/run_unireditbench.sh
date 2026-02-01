GPUS=8

input_path=./UniREditBench
output_path=/output_dir

# Image Editing with Reasoning
torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
   ./gen/gen_images_mp_uniredit.py \
    --input_dir $input_path \
    --output_dir $output_path \
    --metadata_file /inspire/qb-ilm/project/deepgen/public/UniREditBench/data.json \
    --max_latent_size 64 \
    --model-path $model_path \
    --think

