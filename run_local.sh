#!/bin/bash

# 清理之前的Ray进程
echo "🧹 清理Ray环境..."
ray stop --force 2>/dev/null || true
pkill -f ray:: 2>/dev/null || true
sleep 2

# 设置本地单机单卡运行的环境变量
export WORLD_SIZE=1
export RANK=0
export LOCAL_WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 设置CUDA相关
export CUDA_VISIBLE_DEVICES=0

# 设置Ray优化
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_LOCAL_MODE=1

# 运行训练
echo "🚀 启动本地训练模式..."
echo "环境变量设置:"
echo "  WORLD_SIZE=$WORLD_SIZE"
echo "  RANK=$RANK"
echo "  LOCAL_WORLD_SIZE=$LOCAL_WORLD_SIZE"
echo "  LOCAL_RANK=$LOCAL_RANK"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""

cd /data/dlf/code/self-questioning-lm



python -m verl.trainer.main_ppo exps="[grpo,ch_painting,minimal_local]" trainer.experiment_name=debug 