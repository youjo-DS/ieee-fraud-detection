import os
import requests
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import feather

from glob import glob

from socket import gethostname
HOSTNAME = gethostname()

from tqdm import tqdm
from sklearn.model_selection import KFold
from time import time, sleep
from datetime import datetime
from multiprocessing import cpu_count, Pool
import gc

# ==========================================================================
# config
# ==========================================================================
TRAIN_DIR = ''
TEST_DIR  = ''


# ==========================================================================
# function
# ==========================================================================
def start(fname):
    global st_time
    st_time = time()
    print(f"""
#==============================================================================
# START!!! {fname}    PID: {os.getpid()}    time: {datetime.today()}
#==============================================================================
""")
    # send_line(f'{HOSTNAME}  START {fname}  time: {elapsed_minute():.2f}min')
    return

def reset_time():
    global st_time
    st_time = time()
    return

def elapsed_minute():
    return (time() - st_time)/60

def end(fname):
    print(f"""
#==============================================================================
# SUCCESS !!! {fname}
#==============================================================================
""")
    print(f'time: {elapsed_minute():.2f}min')
    send_line(f'{HOSTNAME}  FINISH {fname}  time: {elapsed_minute():.2f}min')
    return

def send_line(message):
    
    line_notify_token = 'W7AWtVJn1EBQ8yTNBhlVvIh8ADZs3FQF25Mj1LG17Tc'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    requests.post(line_notify_api, data=payload, headers=headers)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def save_importances(importances):
    if len(importances['feaure'].nunique()) > 300:
        order = importances.agg({'gain': 'mean'}).sort_values('gain', ascending=False).head(300).index
        plt.figure(figsize=(16, len(order) / 3))
        sns.barplot(x='gain',
                    y='feature',
                    data=importances[importances['feature'].isin(order)].sort_values('gain', ascending=False))
        plt.savefig(f'LOG/imp/PNG/imp_{__file__}.png', dpi=200, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.figure(figsize=(16, int(len(importances) / 3)))
        sns.barplot(x='gain', y='feature', data=importances.sort_values('gain', ascending=False))
        plt.savefig(f'LOG/imp/PNG/imp_{__file__}.png', dpi=200, bbox_inches="tight", pad_inches=0.1)
        
    return

# =============================================================================
# load feature
# =============================================================================
def load_feature(train, test, USE_FEATURE, FEATURE_DIR='../feature/'):
    len_tr = len(train)
    len_te = len(test)

    if isinstance(USE_FEATURE, str):
        USE_FEATURE = [USE_FEATURE]

    if len(USE_FEATURE) > 0:
        tr_files = []
        te_files = []
        for f in USE_FEATURE:
            tr_file   = glob(f'../feature/{f}*__train__*.ftr')
            tr_files += tr_file

            te_file   = glob(f'../feature/{f}*__test__*.ftr')
            te_files += te_file
    else:
        tr_feature_path = '../feature/*__train__*.ftr'
        te_feature_path = '../feature/*__test__*.ftr'

        tr_files = sorted(glob(tr_feature_path))
        te_files = sorted(glob(te_feature_path))

    train = pd.concat([train, *[feather.read_dataframe(f) for f in tr_files]], axis=1)
    test  = pd.concat([test, *[feather.read_dataframe(f) for f in te_files]], axis=1)

    assert len(train) == len_tr
    assert len(test) == len_te

    return train, test