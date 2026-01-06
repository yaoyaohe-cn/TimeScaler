#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Create logs directory
if [ ! -d "./logs/TimeScaler" ]; then
    mkdir -p ./logs/TimeScaler
fi


model_name=TimeScaler
data=ETTh2


seq_lens=(512 512 512 512) 
pred_lens=(96 192 336 720)

seq_lens=(512 512 512 512) 
pred_lens=(96 192 336 720)
wavelets=(db4 db4 db4 db4)
levels=(3 3 3 3) 
d_models=(128 128 128 128)
dropouts=(0.2 0.2 0.2 0.2)
decay=(0.00047377723860348293 0.00047377723860348293 0.00047377723860348293 0.00047377723860348293)
learning_rates=(0.0025 0.0025 0.0025 0.0025)
batches=(256 256 256 256)
epochs=(30 30 30 30)
patiences=(5 5 5 5)

for i in "${!pred_lens[@]}"; do
    log_file="logs/TimeScaler/${data}_${pred_lens[$i]}.log"
    
    echo "Running ${data} Prediction ${pred_lens[$i]}..."
    
    python -u run_LTF.py \
        --model $model_name \
        --data $data \
        --seq_len ${seq_lens[$i]} \
        --pred_len ${pred_lens[$i]} \
        --d_model ${d_models[$i]} \
        --wavelet ${wavelets[$i]} \
        --level ${levels[$i]} \
        --dropout ${dropouts[$i]} \
        --learning_rate ${learning_rates[$i]} \
        --batch_size ${batches[$i]} \
        --train_epochs ${epochs[$i]} \
        --weight_decay ${decay[$i]} \
        --patience ${patiences[$i]} \
        > $log_file
done
