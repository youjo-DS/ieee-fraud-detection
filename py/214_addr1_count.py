import pandas as pd
import numpy as np

import feather
import datetime
from glob import glob

import feature
import utils
import super_aggre

PREF = 'f214'
COUNT_FEATURE = 'addr1'

utils.start(__file__)

DIR = '../input/pickle/'
train = pd.read_pickle(DIR + 'train_df.pkl')
test  = pd.read_pickle(DIR + 'test_df.pkl')

# drop
use_col = ['DT_M', COUNT_FEATURE]
drop_col = [col for col in train.columns if col not in use_col]

train.drop(columns=drop_col, inplace=True)
test.drop(columns=drop_col, inplace=True)

# get index
len_tr = len(train)

# full count
df = pd.concat([train, test], axis=0)

card_count = pd.crosstab(index=df[COUNT_FEATURE],
                         columns=df['DT_M'],
                         normalize='columns')


lags = [0, 1, 2, 3, 4, 5, 6]
for lag in lags:
    df[f'{COUNT_FEATURE}_counts__lag{lag}'] = np.nan
    for num in df['DT_M'].unique():
        if (num - lag) < 12:
            pass
        elif (num - lag) == 18:
            pass
        else:
            df.loc[df['DT_M'] == num, f'{COUNT_FEATURE}_counts__lag{lag}'] = df[df['DT_M'] == num][f'{COUNT_FEATURE}'].map(card_count[num - lag])

df.drop(columns=use_col, inplace=True)

assert len(df.columns) == len(lags)

train = df.iloc[:len_tr]
test  = df.iloc[len_tr:]

train = train.add_prefix(PREF + '__').reset_index(drop=True)
test  = test.add_prefix(PREF + '__').reset_index(drop=True)

assert len(train.columns) == len(lags)

feature.save_feature(train, 'train', PREF, f'{COUNT_FEATURE}_fq_lag')
feature.save_feature(test, 'test', PREF, f'{COUNT_FEATURE}_fg_lag')

utils.end(__file__)

