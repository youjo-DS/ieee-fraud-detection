import pandas as pd
import numpy as np

import feather
import datetime
from glob import glob

import feature
import utils
import super_aggre

from copy import copy
from tqdm import tqdm

PREF = 'f216'

utils.start(__file__)

DIR = '../input/pickle/'
train = pd.read_pickle(DIR + 'train_df.pkl')
test  = pd.read_pickle(DIR + 'test_df.pkl')

tr_cols_raw = train.columns
te_cols_raw = test.columns

len_tr = len(train)
len_te = len(test)

use_cols = ['DT_M', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'isFraud']
drop_cols = [col for  col in test.columns if col not in use_cols]

train.drop(columns=(drop_cols), inplace=True)
test.drop(columns=drop_cols, inplace=True)


stats = ['mean', 'std']
agg = super_aggre.auto_agg(
    data=train,
    group=use_cols,
    agg_cols=['isFraud'],
    agg_funcs=stats
)

lags = [1, 2, 3, 6]
def lag_merger(df, agg, lags):
  for lag in tqdm(lags):
    data = copy(agg)
    data = data.add_suffix(f'__lag{lag}').reset_index()
    data['DT_M'] += lag  
    df = df.merge(data, on=use_cols, how='left')
  return df

train = lag_merger(train, agg, lags)
test  = lag_merger(test, agg, lags)

train.drop(columns=use_cols, inplace=True)
test.drop(columns=use_cols, inplace=True)

for c in train.columns:
  train[c] = train[c].astype('float32')
  test[c]  = test[c].astype('float32')

train = train.add_prefix(PREF + '__')
test = test.add_prefix(PREF + '__')

print(train.columns)
assert len(train.columns) == 8
assert len(test.columns) == 8

feature.save_feature(train, 'train', PREF, 'D9_lag')
feature.save_feature(test, 'test', PREF, 'D9_lag')

utils.end(__file__)

