from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd

class NFTPriceDataset(Dataset):
    def __init__(self, prices_df, tweets_df, encodings, lookback):
        self.projects = prices_df.project.values
        self.dates = prices_df.ts.values
        self.prices = prices_df['mean_norm'].values
        self.prices_og = prices_df['mean'].values

        self.lookback = lookback

        self.tweets = tweets_df[['Unnamed: 0', 'project', 'Datetime', 'LikeCount', 'RetweetCount', 'polarity']]
        self.encodings = encodings

    def __len__(self):
        return len(self.prices)

    def __getitem__(self, index):
        project = self.projects[index]
        dt = self.dates[index]
        price = self.prices[index]

        tweets_tmp = self.tweets[
            (self.tweets['Datetime']<dt) & (self.tweets['project']==project)
        ].sort_values(
            by=['Datetime', 'LikeCount','RetweetCount'], 
            ascending=False
        ).reset_index(drop=True)

        imp_w = None
        ts_w = None
        encs = []

        if len(tweets_tmp) >= self.lookback:
            tweets_tmp = tweets_tmp[:self.lookback]
            imp_w = tweets_tmp.LikeCount.values # have to modify this
            ts_w = 1/(((np.datetime64(dt) - tweets_tmp['Datetime'].values).astype(float)*1e-9)/3600)
            for i, row in tweets_tmp.iterrows():
                encs.append(self.encodings[row['Unnamed: 0']])
        else:
            pad_len = self.lookback - len(tweets_tmp)
            con_list = [tweets_tmp]
            for i in range(pad_len):
                con_list.append(tweets_tmp.iloc[0:1])
            tweets_tmp = pd.concat(con_list)

            tweets_tmp = tweets_tmp[:self.lookback]
            imp_w = tweets_tmp.LikeCount.values # have to modify this
            ts_w = 1/(((np.datetime64(dt) - tweets_tmp['Datetime'].values).astype(float)*1e-9)/3600)
            for i, row in tweets_tmp.iterrows():
                encs.append(self.encodings[row['Unnamed: 0']])

        
        return {
            'encs': torch.tensor(encs, dtype=torch.float),
            'ts_w': torch.tensor(ts_w, dtype=torch.float),
            'imp_w': torch.tensor(imp_w, dtype=torch.float),       
            'price': torch.tensor(price, dtype=torch.float),
            'price_og': torch.tensor(self.prices_og[index], dtype=torch.float)
        }


class NFTMovementDataset(Dataset):
    def __init__(self, prices_df, tweets_df, encodings, lookback):
        self.projects = prices_df.project.values
        self.dates = prices_df.block_timestamp.values
        self.targets = prices_df['label'].values

        self.lookback = lookback

        self.tweets = tweets_df[['Unnamed: 0', 'project', 'Datetime', 'LikeCount', 'RetweetCount', 'polarity']]
        self.encodings = encodings

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        project = self.projects[index]
        dt = self.dates[index]
        target = self.targets[index]

        tweets_tmp = self.tweets[
            (self.tweets['Datetime']<dt) & (self.tweets['project']==project)
        ].sort_values(
            by=['Datetime', 'LikeCount','RetweetCount'], 
            ascending=False
        ).reset_index(drop=True)

        imp_w = None
        ts_w = None
        ts_inv = None
        encs = []

        if len(tweets_tmp) >= self.lookback:
            tweets_tmp = tweets_tmp[:self.lookback]
            imp_w = tweets_tmp.LikeCount.values # have to modify this
            ts_w = ((np.datetime64(dt) - tweets_tmp['Datetime'].values).astype(float)*1e-9)/3600
            ts_inv = 1/ts_w
            for i, row in tweets_tmp.iterrows():
                encs.append(self.encodings[row['Unnamed: 0']])
        elif len(tweets_tmp) == 0:
            imp_w = [0]*self.lookback
            ts_w = [0]*self.lookback
            encs = [[0]*768]*self.lookback
            
        else:
            pad_len = self.lookback - len(tweets_tmp)
            con_list = [tweets_tmp]
            for i in range(pad_len):
                con_list.append(tweets_tmp.iloc[0:1])
            tweets_tmp = pd.concat(con_list)

            tweets_tmp = tweets_tmp[:self.lookback]
            imp_w = tweets_tmp.LikeCount.values # have to modify this
            ts_w = 1/(((np.datetime64(dt) - tweets_tmp['Datetime'].values).astype(float)*1e-9)/3600)
            for i, row in tweets_tmp.iterrows():
                encs.append(self.encodings[row['Unnamed: 0']])

        
        return {
            'encs': torch.tensor(encs, dtype=torch.float),
            'ts_w': torch.tensor(ts_w, dtype=torch.float),
            'imp_w': torch.tensor(imp_w, dtype=torch.float),       
            'target': torch.tensor(target, dtype=torch.float),
        }
