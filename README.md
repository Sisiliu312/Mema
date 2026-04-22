# Mema

Mema is built on top of the LLaVA codebase.

## Contents

- [Install](#install)
- [Dataset](#dataset)
- [Train](#train)
- [Evaluation](#evaluation)

## Install

The installation setup follows LLaVA.

Reference:
- LLaVA GitHub: https://github.com/haotian-liu/LLaVA

Basic install commands:

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Dataset

Dataset preparation follows LLaVA's data guide.
Our 20k training set is randomly sampled from `llava_v1_5_mix665k.json`.

Reference:
- LLaVA GitHub: https://github.com/haotian-liu/LLaVA

## Train

Run the following script to train Mema:

```bash
bash /scripts/v1_5/finetune_task.sh
```

## Evaluation

We follow the evaluation setup of LLaVA.

Reference:
- LLaVA GitHub: https://github.com/haotian-liu/LLaVA


## Acknowledgement

- [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon.
- [LLaVA](https://github.com/haotian-liu/LLaVA): the core multimodal foundation this project is based on.