# Data prepration

We provide data examples for **T2I**, **Editing**, **Reasoning_T2I** ,**Reasoning_Editing** ,**Interleaved_T2I** and **Interleaved_Editing** tasks in **`data/dataset_info.py`**.

The open-source T2I datasets we used are below:
| DATASET        | Download Link                                                |
| ---------- | ------------------------------------------------------------ |
| BLIP3o-60k | https://huggingface.co/datasets/BLIP3o/BLIP3o-60k |
| ShareGPT-4o-Image-T2I | https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image |
| OpenGPT-4o-Image-T2I | https://huggingface.co/datasets/WINDop/OpenGPT-4o-Image |
| Echo-4o-Image | https://huggingface.co/datasets/Yejy53/Echo-4o-Image |

The open-source Editing datasets we used are below:
| DATASET        | Download Link                                                |
| ---------- | ------------------------------------------------------------ |
| Nano-consistent-150k |https://huggingface.co/datasets/Yejy53/Nano-consistent-150k |
| ShareGPT-4o-Image-Editing | https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image |
| OpenGPT-4o-Image-Editing | https://huggingface.co/datasets/WINDop/OpenGPT-4o-Image |
| pico-banana-400k | https://github.com/apple/pico-banana-400k |

**Reasoning_T2I** ,**Reasoning_Editing**  and **Interleaved_T2I** are contained in [Unireason reason tuning data](https://huggingface.co/datasets/Alex11556666/Reason_Tuning)

We have refactored the dataloader code so that data for any task can be passed through JSON files. The specific format is as follows:

T2I data format JSON file:
```
.....
  {
    "type": "T2I_SFT",
    "txt": "Waist-up, a male figure with jet-black hair in cyberpunk-80s attire, his form defined by Renaissance chiaroscuro, intently deciphers hieroglyphs on a massive column under vibrant top-down golden hour light, the column leading the eye within a plein air composition.",
    "image": "",
    "image_path": "image/16685.png"
  },
.....
```

Editing data format JSON file:
```
.....
  {
    "input_image": [
      "editing/input_11656.jpg"
    ],
    "output_image": "editing/output_11656.png",
    "instruction": "Change the image style to Monet's Impressionist style."
  },
.....
```
1. **Download the dataset from link** Follow the official process for each data to extract, then process it into the above JSON file format.

2. Edit every `your_data_path` placeholder in **`data/dataset_info.py`** .
   
3. *(Optional)*  Extend `DATASET_INFO` with your own data JSONL file to mix extra data.

---

# Training

The reason tuing recipe looks like this (replace environment variables with real paths or values with your own):
the stage_1_align checkpoints can download from https://huggingface.co/Alex11556666/UniReason/tree/main/results/stage_1_align

```shell
NUM_NODES=8
NPROC_PER_NODE=8
MODEL_PATH=your_model_path/BAGEL-7B-MoT
PARM_PATH=your_model_path/results/stage_1_align/0030000

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
```


You are encouraged to adjust any of hyperparameters to fit your GPU budget and the scale of your dataset. 


## Model config


| Argument                     | Default                                     | Description                                                     |
| ---------------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| `llm_path`                   | `hf/Qwen2.5-0.5B-Instruct`                  | Language‑model backbone (HuggingFace repo or local folder).     |
| `vae_path`                   | `flux/vae/ae.safetensors`                   | Pre‑trained VAE checkpoint for latent diffusion.                |
| `vit_path`                   | `hf/siglip-so400m-14-980-flash-attn2-navit` | SigLIP ViT used for image understanding.                        |
| `max_latent_size`            | `32`                                        | Maximum latent grid side; defines highest generable resolution. |
| `latent_patch_size`          | `2`                                         | VAE pixels represented by one latent patch.                     |
| `vit_max_num_patch_per_side` | `70`                                        | Max ViT patches per image side after resizing.                  |
| `text_cond_dropout_prob`     | `0.1`                                       | Probability to drop text conditioning while training.           |
| `vae_cond_dropout_prob`      | `0.3`                                       | Dropout on VAE latent inputs.                                   |
| `vit_cond_dropout_prob`      | `0.3`                                       | Dropout on visual features.                                     |

*(See `ModelArguments` for many more options.)*


## Data config


| Argument                    | Default                     | Description                                               |
| --------------------------- | --------------------------- | --------------------------------------------------------- |
| `dataset_config_file`       | `data/configs/example.yaml` | YAML that groups datasets and assigns sampling weights.   |
| `num_workers`               | `4`                         | Background workers per rank for the PyTorch `DataLoader`. |
| `prefetch_factor`           | `2`                         | Batches pre‑fetched by each worker.                       |
| `max_num_tokens_per_sample` | `16384`                     | Skip raw samples longer than this.                        |
| `max_num_tokens`            | `36864`                     | Hard cap for a packed batch (prevents OOM).               |
| `max_buffer_size`           | `50`                        | Overflow buffer length for oversized samples.             |
| `data_seed`                 | `42`                        | Seed for reproducible shuffling and sampling.             |


## Training config

| Argument                               | Default                | Description                                            |
| -------------------------------------- | ---------------------- | ------------------------------------------------------ |
| `total_steps`                          | `500_000`              | Optimiser steps to run.                                |
| `lr`                                   | `1e-4`                 | Peak learning rate after warm‑up.                      |
| `lr_scheduler`                         | `constant`             | Learning‑rate schedule (`constant` or `cosine`).       |
| `warmup_steps`                         | `2000`                 | Linear warm‑up duration.                               |
| `ema`                                  | `0.9999`               | Exponential moving‑average decay for model weights.    |
| `max_grad_norm`                        | `1.0`                  | Gradient‑clipping threshold.                           |
| `save_every`                           | `2000`                 | Checkpoint frequency (steps).                          |
| `visual_gen / visual_und`              | `True`                 | Enable image generation / understanding branches.      |
| `freeze_llm / freeze_vit / freeze_vae` | `False / False / True` | Freeze selected modules to save VRAM or for ablations. |
| `use_flex`                             | `True` (in example)    | Enable FLEX packing for higher GPU utilisation.        |
| `sharding_strategy`                    | `HYBRID_SHARD`         | FSDP sharding mode.                                    |
| `num_shard`                            | `8`                    | Parameter shards per rank in HYBRID mode.              |

**Distributed‑launch environment variables**

| Var                           | Meaning                           |
| ----------------------------- | --------------------------------- |
| `num_nodes` / `node_rank`     | Multi‑node orchestration indices. |
| `nproc_per_node`              | Number of GPUs per node.          |
| `master_addr` / `master_port` | NCCL rendezvous endpoint.         |


## Logging config


| Argument         | Default               | Description                                          |
| ---------------- | --------------------- | ---------------------------------------------------- |
| `results_dir`    | `results`             | Root directory for logs and metrics.                 |
| `checkpoint_dir` | `results/checkpoints` | Checkpoints are saved here.                          |
| `log_every`      | `10`                  | Steps between console / W\&B logs.                   |
| `wandb_project`  | `bagel`               | Weights & Biases project name.                       |
| `wandb_name`     | `run`                 | Run name inside the project.                         |
| `wandb_offline`  | `False`               | Switch to offline mode (logs locally, sync later).   |
| `wandb_resume`   | `allow`               | Resumption policy if an existing run ID is detected. |

> **Tip**  Export `WANDB_API_KEY` before launching if you want online dashboards.
