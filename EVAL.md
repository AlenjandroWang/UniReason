# GenEval


## Evaluation
Directly run `scripts/eval/run_geneval.sh` to evaluate GenEVAL. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- we Set `--think` for native prompt enhancing
- See [GenEval](https://github.com/djghosh13/geneval/tree/main) for original GenEval prompts.


# DPGBench


## Evaluation
Directly run `scripts/eval/run_dpgbench.sh` to evaluate DPGBench. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- See [DPGBench](https://github.com/TencentQQGYLab/ELLA) for original DPGBench prompts.


# WISE
We modify the code in [WISE](https://github.com/PKU-YuanGroup/WISE/tree/main) for faster evaluation and download prompt json files.


## Evaluation
Directly run `scripts/eval/run_wise.sh` to evaluate WISE. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Use `think` for World Knowledge-Enhanced Textual Reasoning.
- `scripts/eval/run_wise_refine.sh` support both World Knowledge-Enhanced Textual Reasoning and Fine-grained Editing-like Visual Refinement through twice thinking and correct initial image




# GEdit-Bench
We adopt the code in [GEdit-Bench](https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/EVAL.md) for evaluation.

## Evaluation

Modify the model path, the output path in `scripts/eval/run_gedit.sh`. Then, run the following command:
```shell
bash script/eval/run_gedit.sh
```



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
- Use `think` for World Knowledge-Enhanced Textual Reasoning.
- `scripts/eval/run_kris_refine.sh` support both World Knowledge-Enhanced Textual Reasoning and Fine-grained Editing-like Visual Refinement through twice thinking and correct initial image








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

# UniGenBench
We create the eval code of [UniGenBench](https://github.com/PKU-YuanGroup/ImgEdit) for faster evaluation.

## Data prepration
Please download the benchmark data from [UniGenBench](https://github.com/CodeGoat24/UniGenBench/blob/main/data/test_prompts_en.csv) and and place it in the `Benchmark` directory.

