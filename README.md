# GNDPO

<small>
On-policy distillation (OPD) has recently emerged as an important post-training paradigm. By using a stronger teacher model to provide dense, fine-grained supervision for sampled trajectories, OPD offers a clear advantage over reinforcement learning with verifiable rewards (RLVR), which typically depends on sparse binary or outcome-based environmental feedback. However, naive token-level distillation can suffer from severe gradient instability, largely due to magnitude misalignment in outlier states. To address this issue, we propose Globally Normalized Distillation Policy Optimization (GNDPO), a practical method that stabilizes optimization by transforming raw KL scores into batch-level relative advantages. This normalization effectively mitigates gradient explosions while retaining the benefits of token-level guidance. Experimental results show that GNDPO substantially improves training robustness and downstream performance across multimodal reasoning tasks.
</small>

---------

## Installation
```bash
conda create -n verl python==3.10
conda activate verl
cd verl/
pip install -e .
```
## Training
```bash
# prepare the data
bash scripts/geo3k.py
```

```
├── data
│   └── geo3k
|       |── train.parquet
│       └── test.parquet
├── verl
└── README.md
```
```bash
# train the model
bash shell/*
```

## Evaluation
We mainly use [VLMEvalkit](https://github.com/open-compass/VLMEvalKit) to evaluate our models. Please refer to their documentation and our model configs for more details.

## Acknowledgments
Our training code is mainly based on [verl-internvl](https://github.com/Weiyun1025/verl-internvl), which is built upon [verl](https://github.com/volcengine/verl). Compared to the original codebase, we have added the algorithmic implementations of both On-Policy Distillation (OPD) and Globally Normalized Distillation Policy Optimization (GNDPO).