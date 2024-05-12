#!/usr/bin/zsh
#SBATCH --job-name=chat_iteval            # Job name
#SBATCH --output=logs/%A-%a.out             # Output file
#SBATCH --partition=boost_usr_prod                  # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4                # Number of tasks (processes) per node
#SBATCH --time=16:00:00                   # Walltime limit (hh:mm:ss)
#SBATCH --mem-per-gpu=32G
#SBATCH --array=1-6

export TOKENIZERS_PARALLELISM=false

source ~/.zshrc
source $FAST/lm_eval/bin/activate

MODELS=( \
    "swap-uniba/LLaMAntino-2-chat-7b-hf-ITA" \
    "swap-uniba/LLaMAntino-2-chat-13b-hf-ITA" \
    "meta-llama/Meta-Llama-3-8B-Instruct" \
    "mistralai/Mistral-7B-Instruct-v0.2" \
    "mii-community/zefiro-7b-sft-ITA" \
    "mii-community/zefiro-7b-dpo-ITA" \
)

MODEL=${MODELS[${SLURM_ARRAY_TASK_ID}]}
BATCH_SIZE=1

module load cuda

srun accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${MODEL},dtype=bfloat16 \
    --tasks ita_eval \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --output_path $FAST/ita_eval_v2/ \
    --use_cache $FAST/ita_eval_v2/cache/ \
    --cache_requests "true"

