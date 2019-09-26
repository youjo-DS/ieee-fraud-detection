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
NROUND = 15000
NFOLD = 5
SEED  = 42
ETA   = 0.005

# =============================================================================
# load data
# =============================================================================
DIR = '../input/feather/'
train = feather.read_dataframe(DIR + 'train.ftr')
test  = feather.read_dataframe(DIR + 'test.ftr')

train.sort_values('TransactionDT', inplace=True)

tr_cols_raw = train.columns
te_cols_raw = test.columns

len_tr = len(train)
len_te = len(test)

# =============================================================================
# load feature
# =============================================================================
def load_feature(train, test):
    FEATURE_DIR = '../feature/'
    # USE_FEATURE = ['f202', 'f203', 'f204', 'f205']
    USE_FEATURE = ['f202']
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

# =============================================================================
# function
# =============================================================================
def encoding(train_df, test_df):
    for col in ['card4', 'card6', 'ProductCD']:
        print('Encoding', col)
        temp_df = pd.concat([train_df[[col]], test_df[[col]]])
        col_encoded = temp_df[col].value_counts().to_dict()   
        train_df[col] = train_df[col].map(col_encoded)
        test_df[col]  = test_df[col].map(col_encoded)
        print(col_encoded)

    for col in ['M1','M2','M3','M5','M6','M7','M8','M9']:
        train_df[col] = train_df[col].map({'T':1, 'F':0})
        test_df[col]  = test_df[col].map({'T':1, 'F':0})

    for col in ['M4']:
        print('Encoding', col)
        temp_df = pd.concat([train_df[[col]], test_df[[col]]])
        col_encoded = temp_df[col].value_counts().to_dict()   
        train_df[col] = train_df[col].map(col_encoded)
        test_df[col]  = test_df[col].map(col_encoded)
        print(col_encoded)

    return train_df, test_df

train, test = encoding(train, test)

def minify_identity_df(df):
  
    df['id_12'] = df['id_12'].map({'Found':1, 'NotFound':0})
    df['id_15'] = df['id_15'].map({'New':2, 'Found':1, 'Unknown':0})
    df['id_16'] = df['id_16'].map({'Found':1, 'NotFound':0})

    df['id_23'] = df['id_23'].map({'TRANSPARENT':4, 'IP_PROXY':3, 'IP_PROXY:ANONYMOUS':2, 'IP_PROXY:HIDDEN':1})

    df['id_27'] = df['id_27'].map({'Found':1, 'NotFound':0})
    df['id_28'] = df['id_28'].map({'New':2, 'Found':1})

    df['id_29'] = df['id_29'].map({'Found':1, 'NotFound':0})

    df['id_35'] = df['id_35'].map({'T':1, 'F':0})
    df['id_36'] = df['id_36'].map({'T':1, 'F':0})
    df['id_37'] = df['id_37'].map({'T':1, 'F':0})
    df['id_38'] = df['id_38'].map({'T':1, 'F':0})

    df['id_34'] = df['id_34'].fillna(':0')
    df['id_34'] = df['id_34'].apply(lambda x: x.split(':')[1]).astype(np.int8)
    df['id_34'] = np.where(df['id_34']==0, np.nan, df['id_34'])
    
    df['id_33'] = df['id_33'].fillna('0x0')
    df['id_33_0'] = df['id_33'].apply(lambda x: x.split('x')[0]).astype(int)
    df['id_33_1'] = df['id_33'].apply(lambda x: x.split('x')[1]).astype(int)
    df['id_33'] = np.where(df['id_33']=='0x0', np.nan, df['id_33'])

    df['DeviceType'].map({'desktop':1, 'mobile':0})

    return df

train = minify_identity_df(train)
test  = minify_identity_df(test)

for col in ['id_33']:
    train[col] = train[col].fillna('unseen_before_label')
    test[col]  = test[col].fillna('unseen_before_label')
    
    le = LabelEncoder()
    le.fit(list(train[col])+list(test[col]))
    train[col] = le.transform(train[col])
    test[col]  = le.transform(test[col])

# =============================================================================
# card1 noise clean
# =============================================================================
valid_card = train['card1'].value_counts()
valid_card = valid_card[valid_card > 10]
valid_card = list(valid_card.index)

train['card1'] = np.where(train['card1'].isin(valid_card), train['card1'], np.nan)
test['card1'] = np.where(test['card1'].isin(valid_card), test['card1'], np.nan)

# =============================================================================
# freq encoding
# =============================================================================

i_cols = ['card1','card2','card3','card5',
          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
          'D1','D2','D3','D4','D5','D6','D7','D8','D9',
          'addr1','addr2',
          'dist1','dist2',
          'P_emaildomain', 'R_emaildomain'
         ]

for col in i_cols:
    vc = pd.concat([train[col], test[col]]).value_counts()
    train[f'f000__{col}__fq_enc'] = train[col].map(vc)
    test[f'f000__{col}__fq_enc'] = test[col].map(vc)

# =============================================================================
# target encoding
# =============================================================================
for col in ['ProductCD', 'M4']:
    tmp_dict = train.groupby(col)['isFraud'].agg(['mean']).reset_index().rename(columns={'mean': f'f000__{col}__target_mean'})
    tmp_dict.index = tmp_dict[col].values
    tmp_dict = tmp_dict[f'f000__{col}__target_mean'].to_dict()
    
    train[f'f000__{col}__target_mean'] = train[col].map(tmp_dict)
    test[f'f000__{col}__target_mean']  = test[col].map(tmp_dict)

# =============================================================================
# encode str columns
# =============================================================================

for col in list(train):
    if train[col].dtype == 'O':
        train[col] = train[col].fillna('__NA__')
        test[col]  = test[col].fillna('__NA__')

        train[col] = train[col].astype(str)
        test[col]  = test[col].astype(str)

        le = LabelEncoder()
        le.fit(list(train[col]) + list(test[col]))
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])

        train[col] = train[col].astype('category')
        test[col]  = test[col].astype('category')

# =============================================================================
# preprocess
# =============================================================================

train, test = load_feature(train, test)

# Model Features
rm_cols = ['TransactionID', 'TransactionDT', 'isFraud']
feature_columns = list(train)
for col in rm_cols:
    if col in feature_columns:
        feature_columns.remove(col)

X_train = train[feature_columns]
y_train = train['isFraud']
test = test[feature_columns]

del train
gc.collect()

# =============================================================================
# data split
# =============================================================================

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


# trn_idx_list, val_idx_list = ts_split(X_train, y_train)
trn_idx, val_idx = my_split(X_train)
trn_idx_list, val_idx_list = ksplit(X_train.iloc[trn_idx], y_train.iloc[trn_idx])

drop_col = [col for col in X_train.columns if 'f202' in col]

X_train.drop(columns=drop_col, inplace=True)
test.drop(columns=drop_col, inplace=True)


# =============================================================================
# params
# =============================================================================

# my tune
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': NROUND,
    'learning_rate': ETA,
    'max_depth': -1,
    'num_leaves': 2**8, # 500
    'min_data_in_leaf': 100,
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
val_x, val_y = X_train.iloc[val_idx], y_train.iloc[val_idx]
for i in range(NFOLD):
    print(f'Fold{i + 1}')
    print(f'train records: {len(trn_idx_list[i])}')
    print(f'valid records: {len(val_idx)}')
    start_time = time()

    trn_x, trn_y = X_train.iloc[trn_idx_list[i]], y_train.iloc[trn_idx_list[i]]
    # val_x, val_y = X_train.iloc[val_idx_list[i]], y_train.iloc[val_idx_list[i]]

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
    oof_preds[val_idx_list[i]] = model.predict(X_train.iloc[val_idx_list[i]], num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_preds / NFOLD
    test_preds += model.predict(test) / NFOLD

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

importances = importances.groupby('feature', as_index=False).agg({'gain': 'mean'}).head(300)
plt.figure(figsize=(16, int(len(importances) / 3)))
sns.barplot(x='gain', y='feature', data=importances.sort_values('gain', ascending=False))
plt.savefig(f'LOG/imp/PNG/imp_{__file__}.png', dpi=200, bbox_inches="tight", pad_inches=0.1)


# =============================================================================
# Save prediction
# =============================================================================
sub = pd.read_csv('../input/sample_submission.csv.zip')
sub['isFraud'] = test_preds
sub.to_csv(f'../output/sub_{__file__}.csv', index=False)

tr_preds = pd.DataFrame({
  'true': y_train.values,
  'preds': oof_preds
})
tr_preds.to_feather(f'LOG/tr_preds/oof_{__file__}.ftr')

utils.end(__file__)