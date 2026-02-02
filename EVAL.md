# GenEval
We modify the code in [GenEval](https://github.com/djghosh13/geneval/tree/main) for faster evaluation.

## Setup
Install the following dependencies:
```shell
pip install open-clip-torch
pip install clip-benchmark
pip install --upgrade setuptools

sudo pip install -U openmim
sudo mim install mmengine mmcv-full==1.7.2

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
```

Download Detector:
```shell
cd ./eval/gen/geneval
mkdir model

bash ./evaluation/download_models.sh ./model
```

## Evaluation
Directly run `scripts/eval/run_geneval.sh` to evaluate GenEVAL. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Set `metadata_file` to `./eval/gen/geneval/prompts/evaluation_metadata.jsonl` for original GenEval prompts.


# WISE
We modify the code in [WISE](https://github.com/PKU-YuanGroup/WISE/tree/main) for faster evaluation.


## Evaluation
Directly run `scripts/eval/run_wise.sh` to evaluate WISE. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.




# GEdit-Bench
We adopt the code in [GEdit-Bench](https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/EVAL.md) for evaluation.

## Evaluation

Modify the model path, the output path, the api key, and the api url in `scripts/eval/run_gedit.sh`. Then, run the following command:
```shell
bash script/eval/run_gedit.sh
```



# IntelligentBench
TBD


# KRIS
We modify the code in [KRIS-Bench](https://github.com/mercurystraw/Kris_Bench) for faster evaluation.

## Data prepration
Please download the benchmark data from [KRIS-Bench](https://huggingface.co/datasets/Liang0223/KRIS_Bench) and and place it in the `KRIS_Bench` directory.

The final directory structure is:
```shell
KRIS_Bench
├── abstract_reasoning
├── anomaly_correction
├── biology
├── chemistry
├── color_change
├── count_change
├── geography
├── humanities
├── mathematics
├── medicine
├── multi-element_composition
├── multi-instruction_execution
├── part_completion
├── physics
├── position_movement
├── practical_knowledge
├── rule-based_reasoning
├── size_adjustment
├── temporal_prediction
└── viewpoint_change
```

## Evaluation
Directly run `scripts/eval/run_kris.sh` to evaluate KRIS-Bench. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Set `$openai_api_key` in `scripts/eval/run_kris.sh` and `your_api_url` in `eval/gen/kris/metrics_xx.py`. The default GPT version is `gpt-4o-2024-11-20`.
- Use `think` for thinking mode.
- We set `cfg_text_scale=4` and `cfg_img_scale=1.5` by default. Additionally, `cfg_renorm_min=0` is specified for CFG Renorm.







# ImgEdit
We modify the code in [ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit) for faster evaluation.

## Data prepration
Please download the benchmark data from [ImgEdit-Bench](https://huggingface.co/datasets/sysuyy/ImgEdit/blob/main/Benchmark.tar) and and place it in the `Benchmark` directory.

The final directory structure is:
```shell
Benchmark
├── hard
├── multiturn
└── singleturn
    ├── judge_prompt.json
    ├── singleturn.json
    ├── animal
    ├── architecture
    ├── clothes
    ├── compose
    ├── daily object
    ├── for_add
    ├── human
    ├── style
    └── transport
```

## Evaluation
Directly run `scripts/eval/run_imgedit.sh` to evaluate ImgEdit-Bench. The output will be saved in `$output_path`.

</pre>
</details>
