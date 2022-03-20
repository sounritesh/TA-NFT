from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd

class NFTPriceDataset(Dataset):
    def __init__(self, prices_df, tweets_df, encodings, lookback):
        self.projects = prices_df.project.values
        self.dates = prices_df.ts.values
        self.prices = prices_df['mean'].values

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
            ts_w = ((np.datetime64(dt) - tweets_tmp['Datetime'].values).astype(float)*1e-9)/60
            for i, row in tweets_tmp.iterrows():
                encs.append(self.encodings[row['Unnamed: 0']])
        else:
            pad_len = self.lookback - len(tweets_tmp)
            con_list = np.array([tweets_tmp.values])
            for i in range(pad_len):
                np.append(con_list, tweets_tmp.iloc[0].values)
            # print(con_list.shape)
            con_list = con_list.squeeze(0)
            imp_w = con_list[:, 3] # have to modify this

            print(type(con_list[:, 2]), con_list[:, 2], con_list[:, 2].shape)
            ts_w = ((np.datetime64(dt) - (np.array([np.datetime64(x) for x in con_list[:, 2]]))).astype(float)*1e-9)/60

            for row in con_list:
                encs.append(self.encodings[row[0]])

        
        return {
            'encs': torch.tensor(encs, dtype=torch.float),
            'ts_w': torch.tensor(ts_w, dtype=torch.float),
            'imp_w': torch.tensor(imp_w, dtype=torch.float),       
            'price': torch.tensor(price, dtype=torch.float)     
        }


class NFTMovementDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass