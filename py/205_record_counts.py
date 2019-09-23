import pandas as pd
import numpy as np

import feather
import datetime
from glob import glob

import feature
import utils
import super_aggre

PREF = 'f205'

utils.start(__file__)


def datetime_converter(df):
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')    
    df = df.assign(
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

    df['year_month'] = df['year'] * 100 + df['month']
    df['year_month_date'] = df['year'] * 10000 + df['month'] * 100 + df['day']

    df['motnthly_count'] = df['year_month'].map(df['year_month'].value_counts())
    df['daily_count'] = df['year_month_date'].map(df['year_month_date'].value_counts())
    
    return df[['motnthly_count', 'daily_count']]


if __name__ == '__main__':
    DIR = '../input/feather/'
    train = feather.read_dataframe(DIR + 'train.ftr')
    test  = feather.read_dataframe(DIR + 'test.ftr')

    train = datetime_converter(train).add_prefix(PREF + '__')
    test  = datetime_converter(test).add_prefix(PREF + '__')

    assert len(train.columns) == 2
    assert len(test.columns) == 2

    feature.save_feature(train, 'train', PREF, 'date_count')
    feature.save_feature(test, 'test', PREF, 'date_count')

    utils.end(__file__)

