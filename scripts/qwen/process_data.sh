
export TF_CPP_MIN_LOG_LEVEL=3


# prompt and response for baselines
PYTHONPATH=. python3 ./tools/process_data.py \
    --data-dir ./data \
    --processed-data-dir ./processed_data \
    --model-path Qwen/Qwen3-0.6B \
    --data-process-workers 4 \
    --max-prompt-length 460 \
    --dev-num 1000 \
    --model-type qwen