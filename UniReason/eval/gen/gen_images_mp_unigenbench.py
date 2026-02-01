# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import copy
import pandas as pd
import argparse
from safetensors.torch import load_file
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import torch
import torch.distributed as dist
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae

from PIL import Image
from modeling.bagel.qwen2_navit import NaiveCache

SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

def load_metadatas_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    metadatas = [
        {
            "prompt_en": row["prompt_en"],
            "index": row["index"],
        }
        for _, row in df.iterrows()
    ]

    return metadatas

def move_generation_input_to_device(generation_input, device):
    # Utility to move all tensors in generation_input to device
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
    return generation_input


def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def generate_image_with_think(
    prompt, num_timesteps=50, cfg_scale=4.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=4.0, num_images=4,resolution=1024,
    max_length=2048, simple_think=False, device=None
):
    h, w = resolution, resolution

    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    newlens = [0] * num_images
    new_rope = [0] * num_images
    
    # system prompt
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[SYSTEM_PROMPT]* num_images,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)  
        
    ##########  cfg
    generation_input_cfg = model.prepare_vae_latent_cfg(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)]* num_images, 
    )
    generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)
    ##########  cfg
    
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt]* num_images,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)  
        
    ########## think
    tmp_past_key_values = copy.deepcopy(past_key_values)
    tmp_generation_input = model.prepare_start_tokens(newlens, new_rope, new_token_ids)
    tmp_generation_input = move_generation_input_to_device(tmp_generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = model.generate_text(
            past_key_values=tmp_past_key_values,
            max_length=max_length,
            do_sample=True,
            temperature=0.3,
            end_token_id=new_token_ids['eos_token_id'],
            **tmp_generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        think_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]  
        
    # print("="*30, "original think", "="*30)
    # print(think_output) 
    if simple_think:
        think_output_list = think_output.split("</think>")
        try:
            if think_output_list[1] != "":
                think_output = think_output_list[1].strip()
        except (IndexError, AttributeError):
            think_output = think_output_list[0].strip()
        print("="*30, "processed think", "="*30)
        print(think_output) 
    ########## think
    
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[think_output]* num_images,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(h, w)]* num_images, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    ########## generate image
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        unpacked_latent = model.generate_image(
            past_key_values=past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_scale, 
            cfg_interval=cfg_interval,
            timestep_shift=timestep_shift,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type="global",
            cfg_text_past_key_values=None,
            cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
            cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
            cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
            cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
            **generation_input,
        )
    
    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, resolution//16, resolution//16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, resolution//8, resolution//8)
        image = vae_model.decode(latent.to(device))
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)
        image_list.append(tmpimage)

    return image_list, think_output


def generate_image(prompt, num_timesteps=50, cfg_scale=10.0, cfg_interval=[0, 1.0], cfg_renorm_min=0., timestep_shift=1.0, num_images=4, resolution=512, device=None):  # 添加device参数
    past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    newlens = [0] * num_images
    new_rope = [0] * num_images

    generation_input, newlens, new_rope = gen_model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt] * num_images,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            past_key_values = gen_model.forward_cache_update_text(past_key_values, **generation_input)

    generation_input = gen_model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        image_sizes=[(resolution, resolution)] * num_images, 
        new_token_ids=new_token_ids,
    )
    generation_input = move_generation_input_to_device(generation_input, device)

    cfg_past_key_values = NaiveCache(gen_model.config.llm_config.num_hidden_layers)
    cfg_newlens = [0] * num_images
    cfg_new_rope = [0] * num_images

    generation_input_cfg = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_newlens,
        curr_rope=cfg_new_rope, 
        image_sizes=[(resolution, resolution)] * num_images,
    )
    generation_input_cfg = move_generation_input_to_device(generation_input_cfg, device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = gen_model.generate_image(
                past_key_values=past_key_values,
                num_timesteps=num_timesteps,
                cfg_text_scale=cfg_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift,
                cfg_text_past_key_values=cfg_past_key_values,
                cfg_text_packed_position_ids=generation_input_cfg["cfg_packed_position_ids"],
                cfg_text_key_values_lens=generation_input_cfg["cfg_key_values_lens"],
                cfg_text_packed_query_indexes=generation_input_cfg["cfg_packed_query_indexes"],
                cfg_text_packed_key_value_indexes=generation_input_cfg["cfg_packed_key_value_indexes"],
                **generation_input,
            )

    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, resolution//16, resolution//16, 2, 2, 16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, resolution//8, resolution//8)
        image = vae_model.decode(latent.to(device))
        tmpimage = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        tmpimage = Image.fromarray(tmpimage)
        image_list.append(tmpimage)

    return image_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images using Bagel model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--csv_path", type=str, required=True, help="csv path containing metadata for each prompt.")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max_latent_size", type=int, default=64)
    parser.add_argument('--model-path', type=str, default='hf/BAGEL-7B-MoT/')
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()
    
    seed = 42
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if rank == 0:
        print(f"Output images are saved in {output_dir}")

    llm_config = Qwen2Config.from_json_file(os.path.join(args.model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    model_state_dict_path = os.path.join(args.model_path, "ema.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    if rank == 0:
        print(msg)
    del model_state_dict

    model = model.to(device).eval()
    vae_model = vae_model.to(device).eval()
    gen_model = model

    cfg_scale = args.cfg_scale
    cfg_interval = [0, 1.0]
    timestep_shift = 3.0
    num_timesteps = 50
    cfg_renorm_min = 0.0

    metadatas = load_metadatas_from_csv(args.csv_path)
    total_metadatas = len(metadatas)

    
    prompts_per_gpu = (total_metadatas + world_size - 1) // world_size
    start = rank * prompts_per_gpu
    end = min(start + prompts_per_gpu, total_metadatas)
    print(f"GPU {rank}: Processing {end - start} prompts (indices {start} to {end - 1})")

    for idx in range(start, end):
        metadata = metadatas[idx]
        prompt = metadata['prompt_en']
        print(f"GPU {rank} processing prompt {idx - start + 1}/{end - start}: '{prompt}'")

   

        image_list = []
        

        for i in range(args.num_images // args.batch_size):
            if args.think:
                tmp_image_list, think_output = generate_image_with_think(
                prompt=prompt,
                cfg_scale=cfg_scale, 
                cfg_interval=cfg_interval, 
                cfg_renorm_min=cfg_renorm_min,
                timestep_shift=timestep_shift, 
                num_timesteps=num_timesteps,
                resolution=args.resolution,
                num_images=args.batch_size,
                max_length=2048, 
                simple_think=True, 
                device=device,
            )
            else:
                tmp_image_list = generate_image(
                    prompt=prompt,
                    cfg_scale=cfg_scale, 
                    cfg_interval=cfg_interval, 
                    cfg_renorm_min=cfg_renorm_min,
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    num_images=args.batch_size,
                    resolution=args.resolution,
                    device=device,
                )

            image_list.extend(tmp_image_list)

        sample_count = 0
        for sample in image_list:
            sample = sample.crop(sample.getbbox())
            sample_path = f"{args.output_dir}/{str(int(metadata['index']))}_{sample_count}.png"
            sample.save(sample_path)
            sample_count += 1

    print(f"GPU {rank} has completed all tasks")
    dist.barrier()