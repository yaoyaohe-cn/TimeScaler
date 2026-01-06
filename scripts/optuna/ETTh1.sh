#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# [FIX] Automatically create the log directory if it doesn't exist
if [ ! -d "./logs/Optuna" ]; then
    mkdir -p ./logs/Optuna
fi

model_name=TimeScaler
data=ETTh1
n_jobs=1
trial_num=50

# Log-uniform LR range (Min Max)
lrs="0.0001 0.005"

pred_lens=(96 192 336 720)

for pred_len in "${pred_lens[@]}"; do
    echo ">>> Running Smart Optuna: $data | Pred: $pred_len"
    
    python -u run_LTF.py \
      --model $model_name \
      --data $data \
      --pred_len $pred_len \
      --use_hyperParam_optim \
      --n_jobs $n_jobs \
      --optuna_trial_num $trial_num \
      --optuna_lr $lrs \
      --train_epochs 30 \
      --patience 5 \
      > logs/Optuna/${data}_${pred_len}.log 2>&1
done
