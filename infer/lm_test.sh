#!/bin/bash
export HF_DATASETS_TRUST_REMOTE_CODE=1
# 1. 环境变量：只需指向你的 SGLang 源码和模型代码
export PYTHONPATH=/home/wangyuxing/wangyuhang/sglang4.1/sglang/python:/home/wangyuxing/sglang-Fairy-plus-minus-i-700M:$PYTHONPATH

export HF_HOME="/home/wangyuxing/.cache/huggingface"


python /home/wangyuxing/wangyuhang/sglang4.1/run_ifairy_sglang.py

