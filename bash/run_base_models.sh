#!/usr/bin/zsh
#SBATCH --job-name=it_eval            # Job name
#SBATCH --output=logs/%A-%a.out             # Output file
#SBATCH --partition=boost_usr_prod                  # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4                # Number of tasks (processes) per node
#SBATCH --time=16:00:00                   # Walltime limit (hh:mm:ss)
#SBATCH --mem-per-gpu=32G
#SBATCH --array=1-11

export TOKENIZERS_PARALLELISM=false

#module load cuda
source ~/.zshrc
source $FAST/lm_eval/bin/activate

MODELS=( \
    "g8a9/tweety-mistral-7b" \
    "swap-uniba/LLaMAntino-2-7b-hf-ITA" \
    "swap-uniba/LLaMAntino-2-13b-hf-ITA" \
    "mii-community/zefiro-7b-base-ITA" \
    "mistralai/Mistral-7B-v0.1" \
    "meta-llama/Llama-2-7b-hf" \
    "meta-llama/Llama-2-13b-hf" \
    "meta-llama/Meta-Llama-3-8B" \
    "sapienzanlp/Minerva-3B-base-v1.0" \
    "sapienzanlp/Minerva-1B-base-v1.0" \
    "sapienzanlp/Minerva-350M-base-v1.0"
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