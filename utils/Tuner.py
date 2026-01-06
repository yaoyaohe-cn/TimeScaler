import torch
import math
from utils.tools import set_random_seed
import optuna
from collections import defaultdict
import pandas as pd
import os
import datetime
from exp.exp_main import Exp_Main

class Tuner:
    # Tuner for TimeScaler Hyperparameter Optimization
    def __init__(self, ranSeed, n_jobs):
        self.fixedSeed = ranSeed
        self.n_jobs = n_jobs 
        self.result_dic = defaultdict(list)
        
        if not os.path.exists('./hyperParameterSearchOutput'):
            os.makedirs('./hyperParameterSearchOutput')
            
        self.current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
    def optuna_objective(self, trial, args):

        # Sequence Length & Architecture
        args.seq_len = trial.suggest_categorical("seq_len", [96, 192, 336, 512])
        args.d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        args.dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        args.wavelet = trial.suggest_categorical("wavelet", ["db4", "sym3", "coif3"])
        args.level = trial.suggest_categorical("level", [2, 3, 4, 5])
            
        args.learning_rate = trial.suggest_loguniform('lr', args.optuna_lr[0], args.optuna_lr[1])
        args.weight_decay = trial.suggest_loguniform('wd', 1e-5, 1e-2)
        

        setting = '{}_{}_sl{}_pl{}_dm{}_lr{}_wd{}_wv{}_lv{}_dp{}_bs{}_sd{}'.format(
            args.model, args.data, args.seq_len, args.pred_len, 
            args.d_model, args.learning_rate, args.weight_decay,
            args.wavelet, args.level, args.dropout, args.batch_size, self.fixedSeed
        )
        
        # Run Experiment
        set_random_seed(self.fixedSeed)
        Exp = Exp_Main
        exp = Exp(args) 
        
        try:
            exp.train(setting, optunaTrialReport=trial)
            test_loss, test_mae = exp.test(setting)
            return test_loss # Minimize Test MSE
            
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            if "out of memory" in str(e).lower():
                print("Trial pruned due to OOM")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            else:
                print(f"Trial failed with error: {e}")
                return float('inf')

    def tune(self, args):
        self.study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.fixedSeed))
        
        self.study.optimize(lambda trial: self.optuna_objective(trial, args), n_trials=args.optuna_trial_num, n_jobs=self.n_jobs) 
        
        self.save_result(args)

    def save_result(self, args):
        # Filename includes Pred_Len so different horizons don't overwrite each other
        file_name = '{}_{}_len{}'.format(args.model, args.data, args.pred_len)
        
        best_params = self.study.best_params
        best_result = self.study.best_value
        
        self.result_dic['data'].append(args.data)
        self.result_dic['pred_len'].append(args.pred_len)
        self.result_dic['best_loss'].append(best_result)
        
        for key, value in best_params.items():
            self.result_dic[key].append(value)
            
        result_df = pd.DataFrame(self.result_dic)
        
        save_path = f'./Output/{file_name}_best_{self.current_time}.csv'
        try:
            result_df.to_csv(save_path, index=False)
            print(f"Optimization results saved to {save_path}")
        except Exception as e:
            print(f'Save failed: {e}')
        print(result_df)
