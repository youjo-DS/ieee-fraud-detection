import gc
import pandas as pd
import numpy as np
import feather

from utils import reduce_mem_usage, start, end

if __name__ == '__main__':
  start(__file__)
  DIR = '../input/'

  tr_tran = pd.read_csv(DIR + 'train_transaction.csv.zip')
  tr_tran = reduce_mem_usage(tr_tran)
  tr_tran.to_feather(DIR + 'feather/train_transaction.ftr')

  del tr_tran
  gc.collect()


  tr_iden = pd.read_csv(DIR + 'train_identity.csv.zip')
  tr_iden = reduce_mem_usage(tr_iden)
  tr_iden.to_feather(DIR + 'feather/train_identity.ftr')

  del tr_iden
  gc.collect()

  te_tran = pd.read_csv(DIR + 'test_transaction.csv.zip')
  te_tran = reduce_mem_usage(te_tran)
  te_tran.to_feather(DIR + 'feather/test_transaction.ftr')

  del te_tran
  gc.collect()


  te_iden = pd.read_csv(DIR + 'test_identity.csv.zip')
  te_iden = reduce_mem_usage(te_iden)
  te_iden.to_feather(DIR + 'feather/test_identidy.ftr')

  del te_iden
  gc.collect()

  end(__file__)