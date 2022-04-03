import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy import test
from prophet import Prophet

from sklearn import preprocessing

from src.data.dataset import NFTPriceDataset, NFTMovementDataset
from src.utils.engine import Engine
import src.model.model as model_pkg
from src.utils.config import DEVICE, UTC
from src.utils.utils import MinMaxScaler

import json
from prophet.serialize import model_to_json, model_from_json

from tqdm import tqdm

def merge_data(prices_df,tweets_df,encodings,lookback):
    projects = prices_df.project.values
    dates = prices_df.ts.values
    prices = prices_df['mean_norm'].values
    tweets = tweets_df[['Unnamed: 0', 'project', 'Datetime', 'LikeCount', 'RetweetCount', 'polarity']]
    tweets["LikeCount"].fillna(0.0, inplace=True)
    tweets["RetweetCount"].fillna(0, inplace=True)
    print (tweets.dtypes)
    likes = []
    rt = []
    polarity = []
    enc = []
    for index in tqdm(range(prices_df.shape[0])):
        project = projects[index]
        dt = dates[index]
        tweets_tmp = tweets[(tweets['Datetime']<dt) & (tweets['project']==project)].sort_values(
            by=['Datetime', 'LikeCount','RetweetCount'], ascending=False).reset_index(drop=True)
        if len(tweets_tmp) >= lookback:
            tweets_tmp = tweets_tmp[:lookback]
        encs = []
        for i, row in tweets_tmp.iterrows():
            encs.append(encodings[row['Unnamed: 0']])
        like_mean = tweets_tmp["LikeCount"].mean()
        rt_mean = tweets_tmp["RetweetCount"].mean()
        polarity_mean = tweets_tmp["polarity"].mean()
        likes.append(like_mean)
        rt.append(rt_mean)
        polarity.append(polarity_mean)
        encs = np.array(encs)
        enc_avg = np.sum(encs, axis=0)
        enc.append(np.mean(enc_avg))
    prices_df.insert(loc=0, column='Likes', value=likes)
    prices_df.insert(loc=1, column='Retweets', value=rt)
    prices_df.insert(loc=2, column='Polarity', value=polarity)
    prices_df.insert(loc=3, column='Enc', value=enc)

    prices_df.rename(columns = {'ts':'ds'}, inplace = True)
    prices_df['ds'] = prices_df['ds'].apply(lambda x: x.replace(tzinfo=None))
    prices_df = prices_df.drop(['mean'], axis = 1)
    prices_df.rename(columns = {'mean_norm':'y'}, inplace = True)

    return prices_df
        

bigdata_folder = os.path.join("/content/drive/MyDrive/NFT-NLP/","dataset/") 
lookback = 5
seed = 42

encodings = np.load(os.path.join(bigdata_folder, "tweet_encodings.npy"))

tweets_ds = pd.read_csv(os.path.join(bigdata_folder, "tweets.csv"))
tweets_ds['Datetime'] = pd.to_datetime(tweets_ds['Datetime']).dt.tz_localize(None)
tweet_scaler1 = MinMaxScaler(tweets_ds['LikeCount'].values, DEVICE)
tweet_scaler2 = MinMaxScaler(tweets_ds['RetweetCount'].values, DEVICE)

# tweets_ds['LikeCount'] = tweet_scaler1.transform(tweets_ds['LikeCount'].values).cpu().numpy()
# tweets_ds['RetweetCount'] = tweet_scaler2.transform(tweets_ds['RetweetCount'].values).cpu().numpy()

prices_ds = pd.read_csv(os.path.join(bigdata_folder, "avg_price.csv"))
prices_ds['ts'] = pd.to_datetime(prices_ds['ts'])
prices_ds = prices_ds.sort_values('ts')

test_size = int(0.15*prices_ds.shape[0])
prices_train = prices_ds[:-test_size]
prices_test = prices_ds[-test_size:]

target_scaler = MinMaxScaler(prices_train['mean'].values, DEVICE)

prices_train['mean_norm'] = target_scaler.transform(prices_train['mean'].values)
prices_test['mean_norm'] = target_scaler.transform(prices_test['mean'].values)

train_ds = merge_data(prices_train, tweets_ds, encodings, lookback)
val_ds = merge_data(prices_test, tweets_ds, encodings, lookback)

train_ds.to_csv("prophet_train.csv")
val_ds.to_csv("prophet_val.csv")

actual_price = val_ds["y"]

val_ds = val_ds.drop(['y'], axis = 1)

print (train_ds.head())
prophet_basic = Prophet()
prophet_basic.add_regressor('Likes')
prophet_basic.add_regressor('Retweets')
prophet_basic.add_regressor('Polarity')
prophet_basic.add_regressor('Enc')


prophet_basic.fit(train_ds)
# with open('prophet_model.json', 'w') as fout:
#     json.dump(model_to_json(prophet_basic), fout)  # Save model


forecast_data = prophet_basic.predict(val_ds)
predicted = forecast_data["yhat"]

se = np.square(predicted - actual_price)
mse = np.mean(se)
print ("MSE: ",mse)