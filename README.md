# ItaEval

[![Leaderboard](https://img.shields.io/badge/Leaderboard-Space-yellow)](https://huggingface.co/spaces/RiTA-nlp/ita-eval)
[![Technical Report](https://img.shields.io/badge/Report-v1-red)](https://bit.ly/itaeval_tweetyita_v1)

This repository contains the configuration and code utilities to run the ItaEval evaluation suite.

- The repository is a fork of the [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). We last aligned on [1980a13](https://github.com/EleutherAI/lm-evaluation-harness/tree/1980a13c9d7bcdc6e2a19228c203f9f7834ac9b8).
- The suite is backing a live leaderboard [HERE](https://huggingface.co/spaces/RiTA-nlp/ita-eval) 

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

## Acknowledgments

ItaEval and TweetyIta are the results of the joint effort of members of the [Risorse per la Lingua Italiana](https://rita-nlp.org/) community. We thank every member that dedicated their personal time to the sprints. We thank CINECA for providing the computational resources (ISCRA grant: HP10C3RW9F).

<!-- ## Cite as

```
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.0},
  doi          = {10.5281/zenodo.10256836},
  url          = {https://zenodo.org/records/10256836}
} 
``` -->
