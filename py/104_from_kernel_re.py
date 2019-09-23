import pandas as pd
import numpy as np

import feather

import feature
import utils

from sklearn.preprocessing import LabelEncoder

PREF = 'f104'

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
    if feature in useful_features:
      # Count encoded for both train and test
      train[f'{PREF}__' + feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
      test[f'{PREF}__' + feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
          
  for feature in ['id_01', 'id_31', 'id_33', 'id_35', 'id_36']:
    if feature in useful_features:
      # Count encoded separately for train and test
      train[f'{PREF}__' + feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
      test[f'{PREF}__' + feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))

  return train, test

if __name__ == '__main__':
  utils.start(__file__)

  DIR = '../input/feather/'
  train = feather.read_dataframe(DIR + 'train.ftr')
  test  = feather.read_dataframe(DIR + 'test.ftr')

  useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',
                  'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                  'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M2', 'M3',
                  'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V17',
                  'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',
                  'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',
                  'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',
                  'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',
                  'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',
                  'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',
                  'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',
                  'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',
                  'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',
                  'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V261',
                  'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',
                  'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',
                  'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 'V323', 'V324', 'V326',
                  'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',
                  'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',
                  'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

  cols_to_drop = [col for col in train.columns if col not in useful_features]
  cols_to_drop.remove('isFraud')
  cols_to_drop.remove('TransactionID')
  cols_to_drop.remove('TransactionDT')

  train = train.drop(cols_to_drop, axis=1)
  test = test.drop(cols_to_drop, axis=1)

  tr_cols = train.columns
  te_cols = test.columns

  train, test = from_kernel(train, test)

  train.drop(columns=tr_cols, inplace=True)
  test.drop(columns=te_cols, inplace=True)

  feature.save_feature(train, 'train', PREF, 'ce_and_day_feature')
  feature.save_feature(test, 'test', PREF, 'ce_and_day_feature')

  utils.end(__file__)