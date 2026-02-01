# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .vlm_dataset import SftJSONLIterableDataset
from .interleave_datasets.think_t2i import ThinkT2IIterableDataset
from .interleave_datasets.t2i import T2ISFTIterableDataset
from .interleave_datasets.think_refine_t2i import ThinkRefineT2IIterableDataset
from .interleave_datasets.edit import TI2ISFTIterableDataset
from .interleave_datasets.think_edit import ThinkEditIterableDataset
from .interleave_datasets.think_refine_edit import ThinkRefineEditIterableDataset
DATASET_REGISTRY = {
    'blip3o_sft_dataset' : T2ISFTIterableDataset,
    'sharegpt4o_sft_dataset': T2ISFTIterableDataset,
    'open4o_sft_dataset': T2ISFTIterableDataset,
    'echo4o_sft_dataset': T2ISFTIterableDataset,
    'share4o_edit_sft' :TI2ISFTIterableDataset,
    'open4o_edit_sft' :TI2ISFTIterableDataset,
    'nano150_edit_sft' :TI2ISFTIterableDataset,
    'picobanana_edit_sft' :TI2ISFTIterableDataset,
    'refine_t2i_long':ThinkRefineT2IIterableDataset,
    't2i_reason':ThinkT2IIterableDataset,
    'edit_reason':ThinkEditIterableDataset,
    "refine_edit":ThinkRefineEditIterableDataset,
}




DATASET_INFO = {   
    'sharegpt4o_sft_dataset': {
        'sharegpt4o_sft_dataset': {
            'jsonl_path': 'your_data_path/ShareGPT-4o-Image/share_4o_img.json',
            'data_dir':'your_data_path/ShareGPT-4o-Image/share_4o_img.json',
            'image_prefix_dir':"your_data_path/ShareGPT-4o-Image/t2i",
        },
    },
    'blip3o_sft_dataset': {
        'blip3o_sft_dataset':{
            'jsonl_path': 'your_data_path/blip3o_60k/blip3o_60k.json',
            'data_dir':'your_data_path/blip3o_60k/blip3o_60k.json',
            'image_prefix_dir' :'your_data_path/blip3o_60k',
        },
    },
    'open4o_sft_dataset': {
        'open4o_sft_dataset':{
            'jsonl_path': 'your_data_path/OpenGPT-4o-Image/OpenGPT-4o-Image.json',
            'data_dir':'your_data_path/OpenGPT-4o-Image/OpenGPT-4o-Image.json',
            'image_prefix_dir' :'your_data_path/OpenGPT-4o-Image/t2i',
        },
    },
    'echo4o_sft_dataset': {
        'echo4o_sft_dataset':{
            'jsonl_path': 'your_data_path/echo-4o-image/echo-4o-image_t2i.json',
            'data_dir':'your_data_path/echo-4o-image/echo-4o-image_t2i.json',
            'image_prefix_dir' :'your_data_path/echo-4o-image',
        },
    },
    'share4o_edit_sft': {
        'share4o_edit_sft':{
            'jsonl_path': 'your_data_path/ShareGPT-4o-Image/text_and_image_to_image.json',
            'data_dir':'your_data_path/ShareGPT-4o-Image/text_and_image_to_image.json',
            'image_prefix_dir':"your_data_path/ShareGPT-4o-Image/editing" ,
        },
    },
    'open4o_edit_sft': {
        'open4o_edit_sft':{
            'jsonl_path': 'your_data_path/OpenGPT-4o-Image/editing.json',
            'data_dir':'your_data_path/OpenGPT-4o-Image/editing.json',
            'image_prefix_dir':"your_data_path/OpenGPT-4o-Image/editing" ,
        },
    },
    'nano150_edit_sft': {
        'nano150_edit_sft':{
            'jsonl_path': 'your_data_path/Nano-150k/data.json',
            'data_dir':'your_data_path/Nano-150k/data.json',
            'image_prefix_dir':"your_data_path/Nano-150k" ,
        },
    },
    'picobanana_edit_sft': {
        'picobanana_edit_sft':{
            'jsonl_path': 'your_data_path/pico-banana-400k/sft_with_local_source_image_path.json',
            'data_dir':'your_data_path/pico-banana-400k/sft_with_local_source_image_path.json',
            'image_prefix_dir':"your_data_path/pico-banana-400k",
        },
    },
    'refine_edit': {
        'refine_edit':{
            'jsonl_path': 'your_data_path/refine_edit/refine_edit.json',
            'data_dir':'your_data_path/refine_edit/refine_edit.json',
            'image_prefix_dir':"your_data_path/refine_edit" ,
        },
    },
    'refine_t2i_long': {
        'refine_t2i_long':{
            'jsonl_path': 'your_data_path/refine_t2i_long/refine_t2i_long.json',
            'data_dir':'your_data_path/refine_t2i_long/refine_t2i_long.json',
            'image_prefix_dir':"your_data_path/refine_t2i_long" ,
        },
    },
    't2i_reason': {
        't2i_reason':{
            'jsonl_path': 'your_data_path/t2i_reason/t2i_reason.json',
            'data_dir':'your_data_path/t2i_reason/t2i_reason.json',
            'image_prefix_dir':"your_data_path/t2i_reason" ,
        },
    },
    'edit_reason': {
        'edit_reason':{
            'jsonl_path': 'your_data_path/edit_reason/edit_reason.json',
            'data_dir':'your_data_path/edit_reason/edit_reason.json',
            'image_prefix_dir':"your_data_path/edit_reason" ,
        },
    }
}