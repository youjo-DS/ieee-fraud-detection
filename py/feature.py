import os
import gc
import feather
from glob import glob

import pandas as pd
import numpy as np

from IPython import embed

from super_aggre import auto_agg, auto_cnt_ratio

import warnings
warnings.filterwarnings('ignore')


from copy import copy


def feature(PREF, FNAME):
    def decorator(func):
        def wrapper(*args, **kwargs):
            dataset = str(kwargs['dataset'])
            ret = func(*args, **kwargs) # function returns Series or DataFrame

            if isinstance(ret, pd.core.series.Series):
                ret = pd.DataFrame(ret)

            ret.columns = [f'{PREF}__{col}' for col in list(ret.columns)]
            save_feature(ret, dataset, PREF, FNAME)
        return wrapper

    return decorator

def save_feature(df: pd.DataFrame, dataset:str, feature_id: str, col_name, with_csv_dump: bool=False):
    path = f'../features/saved/{feature_id}__{dataset}__{col_name}.ftr'
    df.to_feather(path)


def import_feature(PREF, path='../feather/saved/'):
    file = glob(f'{path}{PREF}*.ftr')
    if len(file) == 0:
        print("""
        # ============================================ #
        #       CANNOT FIND FILE OR DIRECTORY!!        #
        # ============================================ #
        """)
    if len(file) > 1:
        print("""
        # ============================================ #
        #         FILE SEQUENCE IS CONFLICT!!          #
        # ============================================ #
        """)

    feature = feather.read_dataframe(file)

    if isinstance(feature, pd.Series):
        feature = pd.DataFrame(feature)

    return feature