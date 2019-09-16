import gc
import pandas as pd
import feather
import utils

if __name__ == '__main__':
  utils.start(__file__)
  DIR = '../input/feather/'
  
  print('# train process...')
  train_transaction = feather.read_dataframe(DIR + 'train_transaction.ftr')
  train_identity    = feather.read_dataframe(DIR + 'train_identity.ftr')

  train = train_transaction.merge(train_identity, on='TransactionID', how='left')
  del train_transaction, train_identity
  gc.collect()

  print('# test process...')
  test_transaction  = feather.read_dataframe(DIR + 'test_transaction.ftr')
  test_identity     = feather.read_dataframe(DIR + 'test_identity.ftr')
  test  = test_transaction.merge(test_identity, on='TransactionID', how='left')
  del test_transaction, test_identity
  gc.collect()

  train.to_feather(DIR + 'train.ftr')
  test.to_feather(DIR + 'test.ftr')

  utils.end(__file__)