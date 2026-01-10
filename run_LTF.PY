import os
import torch
import argparse
import optuna
import numpy as np
import random 
import gc
import time

from exp.exp_main import Exp_Main
from utils.tools import dotdict, set_random_seed
from utils.Tuner import Tuner
from utils.output_database import Output_database

torch.set_printoptions(precision=10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='[TimeScaler] Long Sequence Forecasting')
    
    # Basic Config
    parser.add_argument('--model', type=str, default='TimeScaler', help='model name')
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--data', type=str, default='ETTh1', choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                                                                       'Electricity', 'Traffic', 'Weather'], help='dataset name')
    parser.add_argument('--use_hyperParam_optim', action='store_true', default=False, help='Use Optuna for tuning')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0', help='device ids of multiple gpus')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    
    # TimeScaler Specific Parameters
    parser.add_argument('--seq_len', type=int, default=512, help='input look back window length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction length')
    parser.add_argument('--d_model', type=int, default=256, help='hidden dimension of SPBlocks')
    parser.add_argument('--wavelet', type=str, default='db4', help='wavelet basis (e.g., db2, sym3)')
    parser.add_argument('--level', type=int, default=2, help='decomposition level (J)')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')
    parser.add_argument('--loss', type=str, default='smoothL1', help='loss function: smoothL1 or mse')
    
    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')    
    parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay regularization')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--train_epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--lradj', type=str, default='type3', help='learning rate adjustment strategy')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision')
    
    # Data Loader Parameters
    parser.add_argument('--features', type=str, default='M', help='forecasting task [M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='model checkpoints location')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    
    # Unused but kept for compatibility with legacy scripts if needed
    parser.add_argument('--label_len', type=int, default=0, help='Not used in TimeScaler')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    
    # Optuna Args 
    parser.add_argument('--optuna_trial_num', type=int, default=50, help='Optuna trial number') 
    parser.add_argument('--n_jobs', type=int, default=1, help='optuna jobs')

    parser.add_argument('--optuna_lr', type=float, nargs='+', default=[0.0001, 0.01], help='LR Range (Min, Max)')
    parser.add_argument('--optuna_lradj', type=str, nargs='+', default=['type3'], help='LR Scheduler choices')


    args = parser.parse_args()
    
    # Device Setup
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Data Configuration
    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'Weather': {'data': 'weather.csv', 'root_path': './data/weather/', 'T': 'OT', 'M': [21, 21], 'S': [1, 1], 'MS': [21, 1]},
        'Traffic': {'data': 'traffic.csv', 'root_path': './data/traffic/', 'T': 'OT', 'M': [862, 862], 'S': [1, 1], 'MS': [862, 1]},
        'Electricity': {'data': 'electricity.csv', 'root_path': './data/electricity/', 'T': 'OT', 'M': [321, 321], 'S': [1, 1], 'MS': [321, 1]},
    }
    
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.root_path = data_info['root_path']
        args.target = data_info['T']
        args.c_in = data_info[args.features][0]
        args.c_out = data_info[args.features][1]
    
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    # Standard Execution
    if not args.use_hyperParam_optim:
        print('Args in experiment: {}'.format(args))
        
        # Cleaned Setting String (TimeScaler Specific)
        setting = '{}_{}_sl{}_pl{}_dm{}_lr{}_wd{}_wv{}_lv{}_dp{}_bs{}_sd{}'.format(
            args.model, 
            args.data, 
            args.seq_len, 
            args.pred_len, 
            args.d_model, 
            args.learning_rate,
            args.weight_decay,
            args.wavelet, 
            args.level, 
            args.dropout,
            args.batch_size,   
            args.seed
        )
        
        set_random_seed(args.seed)
        exp = Exp_Main(args) 
        
        print('>>> Start Training: {}'.format(setting))
        exp.train(setting)
        
        print('>>> Start Testing: {}'.format(setting))
        loss_mse, loss_mae = exp.test(setting)
    
    # Optuna Execution
    else:
        print(">>> Starting Optuna Hyperparameter Tuning")
        tuner = Tuner(2025, args.n_jobs)
        tuner.tune(args)
