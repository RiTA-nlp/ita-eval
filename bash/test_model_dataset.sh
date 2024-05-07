#!/usr/bin/zsh
#SBATCH --job-name=it_eval            # Job name
#SBATCH --output=logs/%A-%a.out             # Output file
#SBATCH --partition=boost_usr_prod                  # Specify the partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=1              # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1                # Number of tasks (processes) per node
#SBATCH --time=00:20:00                   # Walltime limit (hh:mm:ss)

export TOKENIZERS_PARALLELISM=false

#module load cuda
source ~/.zshrc
source $FAST/lm_eval/bin/activate

MODEL="g8a9/tweety-mistral-7b"
MODEL="meta-llama/Meta-Llama-3-8B"
MODEL="mii-community/zefiro-7b-base-ITA"
#for MODEL in ${MODELS[@]}; do
echo "Starting $MODEL"
lm_eval --model hf \
    --model_args pretrained=${MODEL},dtype=bfloat16 \
    --tasks gente_rephrasing \
    --batch_size auto \
    --limit 10 \
    --log_samples \
    --output_path $FAST/ita_eval/testing/$MODEL \
    --use_cache $FAST/ita_eval/testing/$MODEL \
    --cache_requests "true"
echo "Done $MODEL"
#done
    # --device cuda:0 \
    # --use_cache $FAST/ita_eval/testing/$MODEL/cache \
        #--limit 50 \
