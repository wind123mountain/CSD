# Code CSD (ICLR 2026) base on DistiLLM (ICML 2024)

<a href="https://arxiv.org/abs/2509.25837"><img src="https://img.shields.io/badge/Paper-arXiv:2402.03898-Green">


## Environment
```bash
bash install.sh
```


## Data Processing
Tokenize the data and store them in binary files:
```bash
bash scripts/qwen/process_data_dolly.sh 
```

## Train CSD

```bash
bash scripts/qwen/train_0.6B_4B_csd_on_policy.sh

bash scripts/qwen/train_0.6B_4B_csd.sh
```


## Run Evaluation
```bash
bash scripts/eval_qwen3_0.6B_distillm.sh
```


