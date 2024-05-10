#!/usr/bin/zsh
#SBATCH --job-name=it_eval_chat            # Job name
#SBATCH --output=logs/%A-%a.out             # Output file
#SBATCH --partition=boost_usr_prod                  # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2                # Number of tasks (processes) per node
#SBATCH --time=16:00:00                   # Walltime limit (hh:mm:ss)
#SBATCH --mem-per-gpu=32G
#SBATCH --array=1

export TOKENIZERS_PARALLELISM=false

source ~/.zshrc
source $FAST/lm_eval/bin/activate

# Positional matching of base model and lora adapters
BASES=( \
    "g8a9/tweety-mistral-7b"
) 
LORAS=( \
    "g8a9/tweety-mistral-7b-sft"
)

BASE=${BASES[${SLURM_ARRAY_TASK_ID}]}
LORA=${LORAS[${SLURM_ARRAY_TASK_ID}]}
BATCH_SIZE=1

module load cuda

accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${BASE},dtype=bfloat16,peft=${LORA} \
    --tasks ita_eval \
    --batch_size $BATCH_SIZE \
    --log_samples \
    --output_path $FAST/ita_eval_v1/$MODEL \
    --use_cache $FAST/ita_eval_v1/$MODEL \
    --cache_requests "true"

