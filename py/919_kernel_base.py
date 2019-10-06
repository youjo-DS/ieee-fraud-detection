import os
os.chdir('../py/')

import sys
import gc
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import feather
import datetime
from glob import glob
from time import time

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

import custom_lgb
import utils
from EDA import *

utils.start(__file__)
# =============================================================================
# set fixed values
# =============================================================================
NLOOP = 300  # 100000
NROUND = 1800
NFOLD = 5  # changed 8 -> 5
SEED  = 42
ETA   = 0.005

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
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# =============================================================================
# load data
# =============================================================================
DIR = '../input/pickle/'
train = pd.read_pickle(DIR + 'train_df.pkl')
test  = pd.read_pickle(DIR + 'test_df.pkl')

remove_features = pd.read_pickle(DIR + 'remove_features.pkl')
remove_features = list(remove_features['features_to_remove'].values)
print('Shape control:', train.shape, test.shape)

train = reduce_mem_usage(train)
tset  = reduce_mem_usage(test)

# type_dict = dict(zip(train.columns, [str(i) for i in train.dtypes]))
# for k, v in type_dict.items():
#     test[k] = test[k].astype(v)

features_columns = [col for col in list(train) if col not in remove_features]

X_train = train[features_columns]
y_train = train['isFraud']
X_test  = test[features_columns]
y_test  = test['isFraud']

sub = test[['TransactionID', 'isFraud']]

del train
gc.collect()

def stratified(X_train, y_train):
    folds = StratifiedKFold(n_splits=NFOLD,
                            shuffle=True,
                            random_state=SEED)
    trn_idx_list = []
    val_idx_list = []
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train.values, y_train.values)):
        trn_idx_list.append(trn_idx)
        val_idx_list.append(val_idx)

    return trn_idx_list, val_idx_list

def ts_split(X_train, y_train):
    folds = TimeSeriesSplit(n_splits=NFOLD)
    trn_idx_list = []
    val_idx_list = []
    for trn_idx, val_idx in folds.split(X_train, y_train):
        trn_idx_list.append(trn_idx)
        val_idx_list.append(val_idx)

    return trn_idx_list, val_idx_list

def ksplit(X_train, y_train):
    folds = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    trn_idx_list = []
    val_idx_list = []
    for trn_idx, val_idx in folds.split(X_train, y_train):
        trn_idx_list.append(trn_idx)
        val_idx_list.append(val_idx)

    return trn_idx_list, val_idx_list

def my_split(X_train):
    trn_idx = X_train[X_train['f202__date_block'] < 201805].index
    val_idx = X_train[X_train['f202__date_block'] >= 201805].index

    return trn_idx, val_idx


trn_idx_list, val_idx_list = ksplit(X_train, y_train)


# =============================================================================
# params
# =============================================================================

# my tune
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'tree_learner': 'serial',
    'metric': 'auc',
    'n_estimators': NROUND,
    'learning_rate': ETA,
    'max_depth': -1,
    'num_leaves': 2**8, # 500
    # 'min_data_in_leaf': 100,
    # 'min_sum_hessian_in_leaf': 10,
    'min_child_weight': 3,
    'bagging_fraction': 0.7,
    'feature_fraction': 0.7,
    'tree_learner': 'serial',
    'subsample_freq': 1,
    # 'lambda_l1': 0.1,
    # 'lambda_l2': 0.01,
    'bagging_seed': SEED,
    'early_stopping_round': 100,
    'verbosity': -1,
}


# =============================================================================
# predict
# =============================================================================

importances = pd.DataFrame()
val_preds = []
oof_preds = np.zeros(len(y_train))
test_preds = np.zeros(len(test))

val_aucs = []
for i in range(NFOLD):
    print(f'Fold{i + 1}')
    print(f'train records: {len(trn_idx_list[i])}')
    print(f'valid records: {len(val_idx_list[i])}')
    start_time = time()

    trn_x, trn_y = X_train.iloc[trn_idx_list[i]], y_train.iloc[trn_idx_list[i]]
    val_x, val_y = X_train.iloc[val_idx_list[i]], y_train.iloc[val_idx_list[i]]

    dtrain = lgb.Dataset(trn_x, label=trn_y)
    dvalid = lgb.Dataset(val_x, label=val_y)
    
    model = lgb.train(
        params,
        train_set=dtrain,
        valid_sets=[dtrain, dvalid],
        num_boost_round=NROUND,
        verbose_eval=500
    )

    val_preds = model.predict(val_x, num_iteration=model.best_iteration)
    oof_preds[val_idx_list[i]] = val_preds
    test_preds += model.predict(X_test) / NFOLD

    val_score = roc_auc_score(val_y, val_preds)
    val_aucs.append(val_score)

    print('\t AUC = ', val_score)

    imp_df = pd.DataFrame({
        'feature': X_train.columns,
        'gain': model.feature_importance(),
        'fold': i + 1
    })
    importances = pd.concat([importances, imp_df], axis=0)

    print(f'Fold {i+1} finished in {str(datetime.timedelta(seconds=time() - start_time))}')

mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(y_train, oof_preds)

print(f'Mean AUC: {mean_auc:.9f}, std: {std_auc:.9f}, All_AUC: {all_auc:.9f}')

# =============================================================================
# Save imp
# =============================================================================
importances.to_csv(f'LOG/imp/csv/imp_{__file__}.csv', index=False)

importances = importances.groupby('feature', as_index=False).agg({'gain': 'mean'})\
                         .sort_values('gain', ascending=False).head(300)
plt.figure(figsize=(16, int(len(importances) / 3)))
sns.barplot(x='gain', y='feature', data=importances.sort_values('gain', ascending=False))
plt.savefig(f'LOG/imp/PNG/imp_{__file__}.png', dpi=200, bbox_inches="tight", pad_inches=0.1)


# =============================================================================
# Save prediction
# =============================================================================
# sub = pd.read_csv('../input/sample_submission.csv.zip')
sub['isFraud'] = test_preds
sub.to_csv(f'../output/sub_{__file__}.csv', index=False)

tr_preds = pd.DataFrame({
  'true': y_train.values,
  'preds': oof_preds
})
tr_preds.to_feather(f'LOG/tr_preds/oof_{__file__}.ftr')

utils.end(__file__)