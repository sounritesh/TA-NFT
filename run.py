from torch.utils.data import DataLoader
import torch
from argparse import ArgumentParser
import optuna
import numpy as np
import pandas as pd
import random
import os
from sklearn import preprocessing

from src.data.dataset import NFTPriceDataset, NFTMovementDataset
from src.utils.engine import Engine
import src.model.model as model_pkg
from src.utils.config import DEVICE, UTC
from src.utils.utils import MinMaxScaler

parser = ArgumentParser(description="Train model on the dataset and evaluate results.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for all sampling purposes")
parser.add_argument("--data_dir", type=str, help="Path to data folder")
# parser.add_argument("--bert_path", default="bert-base-multilingual-uncased", type=str, help="Path to base bert model")

parser.add_argument("--model", default="transformer", type=str, help="Name of the model: [mlp, lstm, tlstm, wtlstm, transformer]")

parser.add_argument("--lr", type=float, default=1e-4, help="Specifies the learning rate for optimizer")
parser.add_argument("--dropout", type=float, default=0.3, help="Specifies the dropout for BERT output")

parser.add_argument("--tune", action="store_true", help="To tune model by trying different hyperparams")
parser.add_argument("--classification", action="store_true", help="To specify whether it is a classification or regression problem")
parser.add_argument("--zero_shot", action="store_true", help="To specify whether to train and evaluate for zero shot learning")

parser.add_argument("--output_dir", type=str, help="Path to output directory for saving model checkpoints")

parser.add_argument("--max_len", type=int, default=128, help="Specifies the maximum length of input sequences")
parser.add_argument("--hidden_size", type=int, default=200, help="Specifies the hidden size of fully connected layer")
parser.add_argument("--lstm_hidden_size", type=int, default=384, help="Specifies the hidden size of LSTM layer")
parser.add_argument("--lookback", type=int, default=5, help="Specifies the lookback period")

parser.add_argument("--epochs", type=int, default=10, help="Specifies the number of training epochs")

parser.add_argument("--train_batch_size", type=int, default=64, help="Specifies the training batch size")
parser.add_argument("--val_batch_size", type=int, default=64, help="Specifies the validation and testing batch size")

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def run_training(params, save_model=False):
    #encodings = np.load(os.path.join(args.data_dir, "tweet_encodings.npy"))

    #tweets_ds = pd.read_csv(os.path.join(args.data_dir, "tweets.csv"))
    #tweets_ds['Datetime'] = pd.to_datetime(tweets_ds['Datetime']).dt.tz_localize(None)
    #tweet_scaler1 = MinMaxScaler(tweets_ds['LikeCount'].values, DEVICE)
    #tweet_scaler2 = MinMaxScaler(tweets_ds['RetweetCount'].values, DEVICE)

    #tweets_ds['LikeCount'] = tweet_scaler1.transform(tweets_ds['LikeCount'].values).cpu().numpy()
    #tweets_ds['RetweetCount'] = tweet_scaler2.transform(tweets_ds['RetweetCount'].values).cpu().numpy()

    
    if args.model == 'mlp':
        model = model_pkg.MLP(params)
    elif args.model == 'lstm':
        model = model_pkg.LSTM_MLP(params)
    elif args.model == 'tlstm':
        model = model_pkg.TimeLSTM_MLP(params)
    elif args.model == 'tlstm_hawkes':
        model = model_pkg.TLSTM_Hawkes(params, args.train_batch_size)
    elif args.model == 'rtlstm_hawkes':
        model = model_pkg.RTLSTM_Hawkes(params, args.train_batch_size)
    elif args.model == 'transformer':
        model = model_pkg.TransformerEncoder(params, args.train_batch_size)
    
    pt = sum(p.numel() for p in model.parameters())
    pt2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(pt, pt2)
    
    model.to(DEVICE)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.Adam(optimizer_parameters, lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    if args.classification:
        eng = Engine(model, optimizer, DEVICE, args.model, classification=args.classification)
    else:
        eng = Engine(model, optimizer, DEVICE, args.model, target_scaler)

    best_loss = np.inf

    early_stopping_iter = 3
    early_stopping_counter = 0

    for epoch in range(args.epochs):
        train_loss, train_metrics = eng.train(train_dl)
        valid_loss, val_metrics = eng.evaluate(val_dl)

        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(), args.output_dir + "best_timelstm_model.bin")
        
        else:
            early_stopping_counter += 1

        if early_stopping_iter < early_stopping_counter:
            break

        scheduler.step()

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Best Loss: {best_loss}")
        print(f"Epoch: {epoch+1}, Train Metrics: {train_metrics}, Valid Metric: {val_metrics}")

    return best_loss

def objective(trial):
    if args.model =='lstm':
        params = {
            'hidden_size': trial.suggest_int('hidden_size', 18, 768),
            'dropout': trial.suggest_uniform('dropout', 0.1, 0.7),
            'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
            # 'bert_path': args.bert_path,
            'lstm_hidden_size': trial.suggest_int('lstm_hidden_size', 18, 768),
            'input_size': 768,
            'ntargets': 1,
            'device': DEVICE,
            'classification': args.classification
        }
    else:
        params = {
            'hidden_size': trial.suggest_int('hidden_size', 18, 768),
            'dropout': trial.suggest_uniform('dropout', 0.1, 0.7),
            'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
            # 'bert_path': args.bert_path,
            'input_size': 768,
            'ntargets': 1,
            'device': DEVICE,
            'classification': args.classification
        }
    return run_training(params, False)

def main():
    if args.tune:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)

        trial_ = study.best_trial
        print(f"\n Best Trial: {trial_.values}, Params: {trial_.params}")

        score = run_training(trial_.params, True)
        print(score)
    else:
        params = {
            'dropout': args.dropout,
            'lr': args.lr,
            # 'bert_path': args.bert_path,
            'input_size': 768,
            'ntargets': 1,
            'hidden_size': args.hidden_size,
            'lstm_hidden_size': args.lstm_hidden_size,
            'device': DEVICE,
            'classification': args.classification
        }

        run_training(params)

if __name__ == "__main__":
    main()
