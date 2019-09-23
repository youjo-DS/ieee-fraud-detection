import pandas as pd
import numpy as np

import feather
import datetime
from glob import glob

import feature
import utils
import super_aggre

PREF = 'f204'

utils.start(__file__)

DIR = '../input/feather/'
train = feather.read_dataframe(DIR + 'train.ftr')
test  = feather.read_dataframe(DIR + 'test.ftr')

tr_cols_raw = train.columns
te_cols_raw = test.columns

# =============================================================================
# load feature
# =============================================================================
FEATURE_DIR = '../feature/'
USE_FEATURE = ['f202']
if len(USE_FEATURE) > 0:
    tr_files = []
    te_files = []
    for f in USE_FEATURE:
        tr_file   = glob(f'../feature/{f}*tr*.ftr')
        tr_files += tr_file

        te_file   = glob(f'../feature/{f}*te*.ftr')
        te_files += te_file
else:
    tr_feature_path = '../feature/*tr*.ftr'
    te_feature_path = '../feature/*te*.ftr'

    tr_files = sorted(glob(tr_feature_path))
    te_files = sorted(glob(te_feature_path))

train = pd.concat([train, *[feather.read_dataframe(f) for f in tr_files]], axis=1)
test  = pd.concat([test, *[feather.read_dataframe(f) for f in te_files]], axis=1)

len_tr = len(train)

use_cols = ['TransactionAmt', 'dist1', 'dist2', 'f202__date_block', 'card1']
drop_cols = [col for  col in test.columns if col not in use_cols]

train.drop(columns=(drop_cols + ['isFraud']), inplace=True)
test.drop(columns=drop_cols, inplace=True)

df = pd.concat([train, test], axis=0)
df['date_block_num'] = df['f202__date_block'].map(dict(zip(df['f202__date_block'].unique(), range(df['f202__date_block'].nunique()))))

card_count = pd.crosstab(index=df['card1'],
                         columns=df['date_block_num'])


lags = [0, 1, 2, 3, 6]
for lag in lags:
    df[f'card1_counts__lag{lag}'] = np.nan
    for num in range(lag, 13):
        df.loc[df['date_block_num'] == num, f'card1_counts__lag{lag}'] = df[df['date_block_num'] == num]['card1'].map(card_count[num - lag])

df.drop(columns=['TransactionAmt', 'card1', 'dist1', 'dist2', 'f202__date_block', 'date_block_num'], inplace=True)

train = df.iloc[:len_tr]
test  = df.iloc[len_tr:]

train = train.add_prefix(PREF + '__').reset_index(drop=True)
test  = test.add_prefix(PREF + '__').reset_index(drop=True)

assert len(train.columns) == 5

feature.save_feature(train, 'train', PREF, 'numeric_lag')
feature.save_feature(test, 'test', PREF, 'numeric_lag')

utils.end(__file__)

