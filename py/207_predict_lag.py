import pandas as pd
import numpy as np

import feather
import datetime
from glob import glob

import feature
import utils
import super_aggre

PREF = 'f207'

utils.start(__file__)

DIR = '../input/feather/'
train = feather.read_dataframe(DIR + 'train.ftr')
test  = feather.read_dataframe(DIR + 'test.ftr')

tr_cols_raw = train.columns
te_cols_raw = test.columns

# =============================================================================
# load feature
# =============================================================================
DIR = '../input/pickle/'
train = pd.read_pickle(DIR + 'train_df.pkl')
test  = pd.read_pickle(DIR + 'test_df.pkl')

pseudo_tr = feather.read_dataframe('LOG/tr_preds/oof_923_addf206feature_from921.py.ftr')
pseudo_te = pd.read_csv('../output/sub_923_addf206feature_from921.py.csv')

train['pseudo_label'] = pseudo_tr['preds'].values
test['pseudo_label']  = pseudo_te['isFraud'].values

len_tr = len(train)
len_te = len(test)

use_cols = ['DT_M', 'pseudo_label']

train = train[use_cols]
test  = test[use_cols]

df = pd.concat([train, test], axis=0)

agg = df.groupby('DT_M').agg({'pseudo_label': 'mean'})

df = df.merge(agg.add_suffix(f'__lag0').reset_index(), on='DT_M', how='left')

def lag_merger(df, agg, lag):
    shift = agg.shift(lag).add_suffix(f'__lag{lag}').reset_index()
    shift['DT_M'] = shift['DT_M'] + 1

    df = df.merge(shift, on='DT_M', how='left')

    return df

lags = [1, 2, 3, 6]
for lag in lags:
    df = lag_merger(df, agg, lag)

print(df['DT_M'].unique())
df.drop(columns=use_cols, inplace=True)

train = df.iloc[:len_tr].reset_index(drop=True)
test  = df.iloc[len_tr:].reset_index(drop=True)


train = train.add_prefix(PREF + '__')
test = test.add_prefix(PREF + '__')
print(train.columns)
assert len([col for col in train.columns if 'lag0' in col]) > 0
assert len(train.columns) == 5

feature.save_feature(train, 'train', PREF, 'predict_lag')
feature.save_feature(test, 'test', PREF, 'predict_lag')

utils.end(__file__)

