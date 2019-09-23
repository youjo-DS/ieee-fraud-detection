import pandas as pd
import numpy as np

import feather

import feature
import utils

PREF = 'f201'

def count_encoding(train, test, columns):
  for c in columns:
    vc = pd.concat([train[c], test[c]]).value_counts()
    full_null = train[c].isnull().sum() + test[c].isnull().sum()

    train[f'{PREF}__{c}_CE'] = train[c].map(vc)
    test[f'{PREF}__{c}_CE'] = test[c].map(vc)

    train[f'{PREF}__{c}_CE'].fillna(full_null, inplace=True)
    test[f'{PREF}__{c}_CE'].fillna(full_null, inplace=True)
  
  return train, test


if __name__ == '__main__':
  utils.start(__file__)

  DIR = '../input/feather/'
  train = feather.read_dataframe(DIR + 'train.ftr')
  test  = feather.read_dataframe(DIR + 'test.ftr')
  
  tr_cols = train.columns
  te_cols = test.columns

  count_cols = [
    "ProductCD","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain",
    "R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9","DeviceType","DeviceInfo","id_12",
    "id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24",
    "id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_35",
    "id_37","id_38"
  ]
  train, test = count_encoding(train, test, columns=count_cols)

  train.drop(columns=tr_cols, inplace=True)
  test.drop(columns=te_cols, inplace=True)
  
  feature.save_feature(train, 'train', PREF, 'myCE')
  feature.save_feature(test, 'test', PREF, 'myCE')

  utils.end(__file__)