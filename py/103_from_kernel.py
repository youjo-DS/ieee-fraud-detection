import pandas as pd
import numpy as np

import feather

import feature
import utils

from sklearn.preprocessing import LabelEncoder

PREF = 'f103'

def from_kernel(train, test):
  # New feature - decimal part of the transaction amount
  train[f'{PREF}__TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
  test[f'{PREF}__TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)

  # Count encoding for card1 feature. 
  # Explained in this kernel: https://www.kaggle.com/nroman/eda-for-cis-fraud-detection
  train[f'{PREF}__card1_count_full'] = train['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))
  test[f'{PREF}__card1_count_full'] = test['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))

  # https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
  train[f'{PREF}__Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)
  test[f'{PREF}__Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)
  train[f'{PREF}__Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24
  test[f'{PREF}__Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24

  # Some arbitrary features interaction
  for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 
                  'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:

    f1, f2 = feature.split('__')
    train[f'{PREF}__' + feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
    test[f'{PREF}__' + feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

    le = LabelEncoder()
    le.fit(list(train[f'{PREF}__' + feature].astype(str).values) + list(test[f'{PREF}__' + feature].astype(str).values))
    train[f'{PREF}__' + feature] = le.transform(list(train[f'{PREF}__' + feature].astype(str).values))
    test[f'{PREF}__' + feature] = le.transform(list(test[f'{PREF}__' + feature].astype(str).values))
      
  for feature in ['id_34', 'id_36']:
    # Count encoded for both train and test
    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
          
  for feature in ['id_01', 'id_31', 'id_33', 'id_35', 'id_36']:
    # Count encoded separately for train and test
    train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
    test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))

  return train, test

if __name__ == '__main__':
  utils.start(__file__)

  DIR = '../input/feather/'
  train = feather.read_dataframe(DIR + 'train.ftr')
  test  = feather.read_dataframe(DIR + 'test.ftr')

  tr_cols = train.columns
  te_cols = test.columns

  train, test = from_kernel(train, test)

  train.drop(columns=tr_cols, inplace=True)
  test.drop(columns=te_cols, inplace=True)

  feature.save_feature(train, 'train', PREF, 'ce_and_day_feature')
  feature.save_feature(test, 'test', PREF, 'ce_and_day_feature')

  utils.end(__file__)