export TF_CPP_MIN_LOG_LEVEL=3

PYTHONPATH=. python3 ./tools/process_data_dolly.py \
    --data-dir ./data/dolly/ \
    --processed-data-dir ./processed_data/dolly \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --data-process-workers 32 \
    --max-prompt-length 128 \
    --dev-num 1000 \
    --model-type qwen
