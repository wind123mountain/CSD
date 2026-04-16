export TF_CPP_MIN_LOG_LEVEL=3


PYTHONPATH=. python ./tools/process_data_math.py \
    --processed-data-dir ./processed_data/MetaMathQA-50k \
    --model-path Qwen/Qwen2.5-Math-1.5B-Instruct \
    --data-process-workers 64 \
    --max-prompt-length 256 \
    --model-type qwen