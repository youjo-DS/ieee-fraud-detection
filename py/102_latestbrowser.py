import pandas as pd
import numpy as np

import feather

import feature
import utils

PREF = 'f102'


def setbrowser(df):
  df[f'{PREF}__latest_browser'] = np.zeros(len(df))

  df.loc[df["id_31"]=="samsung browser 7.0",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="opera 53.0",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="mobile safari 10.0",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="google search application 49.0",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="firefox 60.0",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="edge 17.0",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 69.0",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 67.0 for android",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 63.0 for android",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 63.0 for ios",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 64.0",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 64.0 for android",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 64.0 for ios",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 65.0",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 65.0 for android",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 65.0 for ios",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 66.0",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 66.0 for android",f'{PREF}__lastest_browser']=1
  df.loc[df["id_31"]=="chrome 66.0 for ios",f'{PREF}__lastest_browser']=1
  
  return df

if __name__ == '__main__':
  utils.start(__file__)

  DIR = '../input/feather/'
  train = feather.read_dataframe(DIR + 'train.ftr')
  test  = feather.read_dataframe(DIR + 'test.ftr')

  train = setbrowser(train)
  test  = setbrowser(test)

  feature.save_feature(train[f'{PREF}__latest_browser'].to_frame(), 'train', PREF, 'latest_browser')
  feature.save_feature(test[f'{PREF}__latest_browser'].to_frame(), 'test', PREF, 'latest_browser')

  utils.end(__file__)