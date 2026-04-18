#! /bin/bash

SEED=42

BASE_PATH=.
MODEL_PATH="./results/qwen2/nnm_0.5B_1.5B_math/3062"
OUTPUT_DIR="${BASE_PATH}/eval_outputs/${MODEL_PATH}-distill"


mkdir -p ${OUTPUT_DIR}

OPTS=""
OPTS=" --gsm8k --math500"
OPTS+=" --data_dir processed_data/MetaMathQA-50k/qwen/"
OPTS+=" --val_batch_size 64"

# devices
OPTS+=" --student_device cuda:0"

# models
OPTS+=" --model_type qwen"
OPTS+=" --output_dir ${OUTPUT_DIR}"

OPTS+=" --bf16"
OPTS+=" --seed ${SEED}"
OPTS+=" --model_path ${MODEL_PATH}"
# OPTS+=" --lora_path "results/qwen2/nnm_0.5B_1.5B_math"
OPTS+=" --tokenizer Qwen/Qwen2.5-0.5B"

python run_eval.py ${OPTS}
# python run_eval.py ${OPTS} >> ${OUTPUT_DIR}/eval.log
