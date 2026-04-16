#! /bin/bash/train_0.6B_4B_csd_on_policy.sh


GPUS=(0)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${#GPUS[@]}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=.
CKPT_NAME="qwen2.5-0.5b"
CKPT="Qwen/Qwen2.5-0.5B"
TEACHER_CKPT_NAME="qwen2.5-1.5b-math"
TEACHER_CKPT="Qwen/Qwen2.5-1.5B-Math"
# data
DATA_DIR="${BASE_PATH}/processed_data/MetaMathQA-50k/qwen/"
# hp
BATCH_SIZE=2
LR=0.0001
GRAD_ACC=8
EVAL_BATCH_SIZE=32
EPOCHS=1
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen2/nnm_0.5B_1.5B_math"
# seed
SEED=42


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --model-type qwen"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num -1"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio 1.0"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval 300"
OPTS+=" --log-interval 20"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_bf16.json"
# type
OPTS+=" --type adaptive-srkl"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 0.95"
OPTS+=" --temperature 0.5"
# distillm
OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"

# OPTS+=" --peft lora"
# OPTS+=" --peft-lora-r 8"
# OPTS+=" --peft-lora-alpha 64"
# OPTS+=" --peft-lora-dropout 0.1"

# # NNM core
# OPTS+=" --lambda-nnm 0.10"
# OPTS+=" --nnm-warmup 100"
# OPTS+=" --K-centroids 128"
# OPTS+=" --d-prime 256"
# OPTS+=" --eta-centroid 0.05"
# OPTS+=" --T-dead 50"
# OPTS+=" --sigma-layer 0.15"
# OPTS+=" --n-mid-layers 4"
# OPTS+=" --ns-iters 5"

# # Teacher correction
# OPTS+=" --do-teacher-correction"
# OPTS+=" --tc-lambda 0.10"
# OPTS+=" --tc-steps 1"

# # Token filtering
# OPTS+=" --high-ent-rho 0.2"

# # Top-K logit KD
# OPTS+=" --top-k-logits 20"

# # Difficulty-aware weighting
# OPTS+=" --use-difficulty-weight"
# OPTS+=" --difficulty-early-layer 2"

# # Temperature annealing
# OPTS+=" --kl-temp-max 3.0"
# OPTS+=" --kl-temp-min 1.0"

# # Centroid pre-pass
# OPTS+=" --centroid-prepass-batches 3000"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}

# ${CMD} \
# >> ${SAVE_PATH}/train.log 2>&1 &
