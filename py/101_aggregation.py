import pandas as pd
import numpy as np

import feather

import feature
import utils

utils.start(__file__)

PREF = 'f101'

DIR = '../input/feather/'

train = feather.read_dataframe(DIR + 'train.ftr')
test  = feather.read_dataframe(DIR + 'test.ftr')

tr_cols = train.columns
te_cols = test.columns

train[f'{PREF}__TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
train[f'{PREF}__TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
train[f'{PREF}__TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
train[f'{PREF}__TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')

test[f'{PREF}__TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
test[f'{PREF}__TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
test[f'{PREF}__TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
test[f'{PREF}__TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')

train[f'{PREF}__id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
train[f'{PREF}__id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
train[f'{PREF}__id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
train[f'{PREF}__id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

test[f'{PREF}__id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
test[f'{PREF}__id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
test[f'{PREF}__id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
test[f'{PREF}__id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')

train[f'{PREF}__D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
train[f'{PREF}__D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train[f'{PREF}__D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
train[f'{PREF}__D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test[f'{PREF}__D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
test[f'{PREF}__D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test[f'{PREF}__D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
test[f'{PREF}__D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

train[f'{PREF}__D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
train[f'{PREF}__D15_to_mean_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('mean')
train[f'{PREF}__D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
train[f'{PREF}__D15_to_std_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('std')

test[f'{PREF}__D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
test[f'{PREF}__D15_to_mean_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('mean')
test[f'{PREF}__D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')
test[f'{PREF}__D15_to_std_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('std')

train.drop(columns=tr_cols, inplace=True)
test.drop(columns=te_cols, inplace=True)

# save
feature.save_feature(train, 'train', PREF, 'tranAmt-id02-D15')
feature.save_feature(test, 'test', PREF, 'tranAmt-id02-D15')

utils.end(__file__)