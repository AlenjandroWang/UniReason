# GenEval


## Evaluation
Directly run `scripts/eval/run_geneval.sh` to evaluate GenEVAL. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- we Set `--think` for native prompt enhancing.
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
- `scripts/eval/run_wise_refine.sh` support both World Knowledge-Enhanced Textual Reasoning and Fine-grained Editing-like Visual Refinement through twice thinking and correct initial image.





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


## Evaluation
Directly run `scripts/eval/run_kris.sh` to evaluate KRIS-Bench. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Use `think` for World Knowledge-Enhanced Textual Reasoning.
- `scripts/eval/run_kris_refine.sh` support both World Knowledge-Enhanced Textual Reasoning and Fine-grained Editing-like Visual Refinement through twice thinking and correct initial image.









# ImgEdit
We modify the code in [ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit) for faster evaluation.

## Data prepration
Please download the benchmark data from [ImgEdit-Bench](https://huggingface.co/datasets/sysuyy/ImgEdit/blob/main/Benchmark.tar) and and place it in the `Benchmark` directory.

## Evaluation
Directly run `scripts/eval/run_imgedit.sh` to evaluate ImgEdit-Bench. The output will be saved in `$output_path`.

#  UniGenBench
We create the eval code of [UniGenBench](https://github.com/CodeGoat24/UniGenBench) for faster evaluation.

## Data prepration
Please download the benchmark data from [UniGenBench](https://github.com/CodeGoat24/UniGenBench/blob/main/data/test_prompts_en.csv) and and place it in the `Benchmark` directory.

## Evaluation
Directly run `scripts/eval/run_unigenbench.sh` to evaluate UniGenBench. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- `scripts/eval/run_unigenbench_refine.sh` support Fine-grained Editing-like Visual Refinement improve initial image quality from object presence, attribute accuracy, style consistency,  and aesthetic quality

# T2I-CoreBench
We create the eval code of [T2I-CoreBench](https://t2i-corebench.github.io/) for faster evaluation.

## Data prepration
Please download the benchmark data from [T2I-CoreBench](https://huggingface.co/datasets/lioooox/T2I-CoReBench) and and place it in the `Benchmark` directory.

## Evaluation
Directly run `scripts/eval/run_corebench.sh` to evaluate T2I-CoreBench. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- `scripts/eval/run_corebench_refine.sh` support Fine-grained Editing-like Visual Refinement improve initial image quality from object presence, attribute accuracy, style consistency,  and aesthetic quality


# UniREditBench
We merge the eval code of [UniREditBench](https://maplebb.github.io/UniREditBench/) for faster evaluation.

## Data prepration
Please download the benchmark data from [UniREditBench](https://huggingface.co/datasets/maplebb/UniREditBench) and and place it in the `Benchmark` directory.

## Evaluation
Directly run `scripts/eval/run_unireditbench.sh` to evaluate UniREditBench. The output will be saved in `$output_path`.
- Set `$model_path` and `$output_path` for the path for checkpoint and log.
- Use `think` for World Knowledge-Enhanced Textual Reasoning.

