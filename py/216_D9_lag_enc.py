import pandas as pd
import numpy as np

import feather
import datetime
from glob import glob

import feature
import utils
import super_aggre

PREF = 'f216'

utils.start(__file__)

DIR = '../input/pickle/'
train = pd.read_pickle(DIR + 'train_df.pkl')
test  = pd.read_pickle(DIR + 'test_df.pkl')

tr_cols_raw = train.columns
te_cols_raw = test.columns

len_tr = len(train)
len_te = len(test)

use_cols = ['DT_M', 'D9']
drop_cols = [col for  col in test.columns if col not in use_cols]

train.drop(columns=(drop_cols), inplace=True)
test.drop(columns=drop_cols, inplace=True)


stats = ['min', 'mean', 'max', 'std']
agg = super_aggre.auto_agg(
    data=train,
    group='DT_M',
    agg_cols=['D9'],
    agg_funcs=stats
)

train = train.merge(agg.add_suffix(f'__lag0'), on='DT_M', how='left')
test  = test.merge(agg.add_suffix(f'__lag0'), on='DT_M', how='left')

def lag_merger(df, agg, lag):
    shift = agg.shift(lag).add_suffix(f'__lag{lag}').reset_index()
    df = df.merge(shift, on='DT_M', how='left')

    return df

lags = [1, 2, 3, 6]
for lag in lags:
    train = lag_merger(train, agg, lag)
    test  = lag_merger(test, agg, lag)

train.drop(columns=use_cols, inplace=True)
test.drop(columns=use_cols, inplace=True)

train = train.add_prefix(PREF + '__')
test = test.add_prefix(PREF + '__')

print(train.columns)
assert len([col for col in train.columns if 'lag0' in col]) > 0
assert len(train.columns) == 20
assert len(test.columns) == 20

feature.save_feature(train, 'train', PREF, 'D9_lag')
feature.save_feature(test, 'test', PREF, 'D9_lag')

utils.end(__file__)

