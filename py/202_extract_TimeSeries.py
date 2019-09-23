import pandas as pd
import numpy as np

import feather
import datetime

import feature
import utils

PREF = 'f202'

def datetime_converter(df):
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')    
    df = df.assign(
            # New feature - decimal part of the transaction amount
            TransactionAmt_decimal = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int),

            # Count encoding for card1 feature. 
            # Explained in this kernel: https://www.kaggle.com/nroman/eda-for-cis-fraud-detection
            card1_count_full = df['card1'].map(df['card1'].value_counts(dropna=False)),

            # https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
            Transaction_day_of_week = np.floor((df['TransactionDT'] / (3600 * 24) - 1) % 7),
            Transaction_hour = np.floor(df['TransactionDT'] / 3600) % 24,

            TransactionDT = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x))),
        )
    df = df.assign(
            # Time of Day
            year = df['TransactionDT'].dt.year,
            month = df['TransactionDT'].dt.month,
            dow = df['TransactionDT'].dt.dayofweek,
            quarter = df['TransactionDT'].dt.quarter,
            hour = df['TransactionDT'].dt.hour,
            day = df['TransactionDT'].dt.day,

            # All NaN
            all_group_nan_sum = df.isnull().sum(axis=1) / df.shape[1],
            all_group_0_count = (df == 0).astype(int).sum(axis=1) / (df.shape[1] - df.isnull().sum(axis=1))
    )

    df['date_block'] = df['year'] * 100 + df['month']
    
    return df

if __name__ == '__main__':
    utils.start(__file__)

    DIR = '../input/feather/'
    train = feather.read_dataframe(DIR + 'train.ftr')
    test  = feather.read_dataframe(DIR + 'test.ftr')

    tr_cols_raw = train.columns
    te_cols_raw = test.columns

    train = datetime_converter(train)
    test  = datetime_converter(test)

    train.drop(columns=tr_cols_raw, inplace=True)
    test.drop(columns=te_cols_raw, inplace=True)

    train = train.add_prefix(PREF + '__')
    test = test.add_prefix(PREF + '__')

    feature.save_feature(train, 'train', PREF, 'ts_feature')
    feature.save_feature(test, 'test', PREF, 'ts_feature')

    utils.end(__file__)

