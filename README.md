# ItaEval

[![Webpage](https://img.shields.io/badge/Webpage-url-blue)](https://bit.ly/tweetyita-itaeval)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Space-yellow)](https://huggingface.co/spaces/RiTA-nlp/ita-eval)
[![Paper](https://img.shields.io/badge/Paper-CLiC_it-red)](https://clic2024.ilc.cnr.it/wp-content/uploads/2024/11/6_main_long.pdf)

This repository contains the configuration and code utilities to run the ItaEval evaluation suite.

- The repository is a fork of the [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). We last aligned on [1980a13](https://github.com/EleutherAI/lm-evaluation-harness/tree/1980a13c9d7bcdc6e2a19228c203f9f7834ac9b8).

## Exploring the Suite 

All the configuration file are under `lm_eval/tasks/ita_eval`. We also have included it as a "benchmark" under `lm_eval/tasks/benchmarks`.

## Getting Started

We release several runner bash scripts to run base and chat models against the suite. Head to `bash/` to find them.

Note that the recipes listed in the folder are tailored to our hardware and you will very likely need to adapt them to yours.

### Run your own model

In a scenario where all of the dependencies are installed correctly, you should be able to run your model on ItaEval with

```bash
MODEL="your-model-id-on-the-huggingface-hub"
lm_eval --model hf \
    --model_args pretrained=${MODEL},dtype=bfloat16 \
    --tasks ita_eval \
    --batch_size 1 \
    --log_samples \
    --output_path "."
```

### Add a model to the [Leaderboard](https://huggingface.co/spaces/RiTA-nlp/ita-eval)

Follow these steps:
1. Run the evaluation with the code above. You will end up with a folder containing a file starting with `results_`
2. Copy and push that folder into this directory: https://huggingface.co/datasets/RiTA-nlp/ita-eval-results/
3. Edit the [model_info.yaml](https://huggingface.co/datasets/RiTA-nlp/ita-eval-results/blob/main/model_info.yaml) file to add the information about the new model(s)
4. Run [this script](https://huggingface.co/datasets/RiTA-nlp/ita-eval-results/blob/main/add_model_info.py) from the main directory of the `ita-eval-results` repository.
5. Push the changes.

Note, points 2 through 5 require having access to the results repository.

## Acknowledgments

ItaEval and TweetyIta are the results of the joint effort of members of the [Risorse per la Lingua Italiana](https://rita-nlp.org/) community. We thank every member that dedicated their personal time to the sprints. We thank CINECA for providing the computational resources (ISCRA grant: HP10C3RW9F).

## Cite
```bibtex
@inproceedings{attanasio2024itaeval,
  title={ItaEval and TweetyIta: A New Extensive Benchmark and Efficiency-First Language Model for Italian},
  author={Attanasio, Giuseppe and Delobelle, Pieter and La Quatra, Moreno and Santilli, Andrea and Savoldi, Beatrice},
  booktitle={CLiC-it 2024: Tenth Italian Conference on Computational Linguistics, Date: 2024/12/04-2024/12/06, Location: Pisa, Italy},
  year={2024}
}
```
