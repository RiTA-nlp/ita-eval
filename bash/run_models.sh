#!/bin/bash
#SBATCH --job-name=lm_eval
#SBATCH --output=./logs/%A-%a.out
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --qos=gpu-medium
#SBATCH --partition=a6000
#SBATCH --array=0-22

export TOKENIZERS_PARALLELISM=false

source ~/mydata/venvs/eval_harness/bin/activate
echo `whereis python`

CHAT_MODELS=( \
    "RiTA-nlp/tweety-Mistral-7B-v0.1-italian-sft-uf_ita" \
    "RiTA-nlp/llama3-tweety-8b-italian-sft-tagengo-merged" \
    "RiTA-nlp/llama3-tweety-8b-italian-sft-uf_ita-merged" \
    "mistralai/Mistral-7B-Instruct-v0.2" \
    "mistralai/Mistral-7B-Instruct-v0.3" \
    "meta-llama/Meta-Llama-3-8B-Instruct" \
    "meta-llama/Llama-3.1-8B-Instruct" \
    "meta-llama/Llama-3.2-1B-Instruct" \
    "meta-llama/Llama-3.2-3B-Instruct" \
    "swap-uniba/LLaMAntino-2-chat-7b-hf-ITA" \
    "swap-uniba/LLaMAntino-2-chat-13b-hf-ITA" \
    "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA" \
    "mii-community/zefiro-7b-sft-ITA" \
    "mii-community/zefiro-7b-dpo-ITA" \
    "mii-llm/maestrale-chat-v0.4-beta" \
    "sapienzanlp/Minerva-7B-instruct-v1.0" \
    "iGeniusAI/Italia-9B-Instruct-v0.1" \
    "CohereForAI/aya-expanse-8b" \
    "utter-project/EuroLLM-1.7B-Instruct" \
    "utter-project/EuroLLM-9B-Instruct" \
    "DeepMount00/Llama-3-8b-Ita" \
    "mudler/Minerva-3B-Llama3-Instruct-v0.1" \
    "mudler/Asinello-Minerva-3B-v0.1" \
)

MODELS=${CHAT_MODELS}
MODEL=${MODELS[${SLURM_ARRAY_TASK_ID}]}

hf_org="hub_results_org=RiTA-nlp"
hf_details_repo="details_repo_name=itaeval-results"
hf_results_repo="results_repo_name=itaeval-results"
push_results="push_results_to_hub=True"
push_samples="push_samples_to_hub=True"
public_repo="public_repo=True"
poc="point_of_contact=giuseppeattanasio6@gmail.com"
gated="gated=True"

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${MODEL},dtype=bfloat16 \
    --tasks itaeval_nlu,itaeval_cfk,itaeval_bfs \
    --batch_size 1 \
    --apply_chat_template \
    --log_samples \
    --output_path ~/myscratch/RiTA/ita_eval_v2/ \
    --use_cache ~/myscratch/RiTA/ita_eval_v2/cache/${MODEL//\//__} \
    --cache_requests "true" \
    --seed 42 \
    --hf_hub_log_args ${hf_org},${hf_details_repo},${hf_results_repo},${push_results},${push_samples},${public_repo},${poc},${gated}

touch logs/ita_eval_v2/${MODEL//\//__}.done

BASE_MODELS=( \
    "RiTA-nlp/tweety-Mistral-7B-v0.1-italian" \
    "RiTA-nlp/llama3-tweety-8b-italian" \
    "mistralai/Mistral-7B-v0.1" \
    "meta-llama/Llama-2-7b-hf" \
    "meta-llama/Llama-2-13b-hf" \
    "meta-llama/Meta-Llama-3-8B" \
    "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
    "swap-uniba/LLaMAntino-2-13b-hf-ITA" \
    "mii-community/zefiro-7b-base-ITA" \
    "sapienzanlp/Minerva-350M-base-v1.0" \
    "sapienzanlp/Minerva-1B-base-v1.0" \
    "sapienzanlp/Minerva-3B-base-v1.0" \
)