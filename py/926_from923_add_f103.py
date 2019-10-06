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
NROUND = 200000
NFOLD = 5
SEED  = 42
ETA   = 0.005

# =============================================================================
# for save model
# =============================================================================

name = f'{__file__}'[:-3]

if not os.path.isdir(f'model/{name}'):
    os.mkdir(f'model/{name}')

save_model_dir = f'model/{name}'

# =============================================================================
# functions
# =============================================================================

# split
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
    trn_idx = X_train[X_train['DT_M'] < 17].index
    val_idx = X_train[X_train['DT_M'] == 17].index

    return trn_idx, val_idx

# =============================================================================
# load data
# =============================================================================
DIR = '../input/pickle/'
train = pd.read_pickle(DIR + 'train_df.pkl')
test  = pd.read_pickle(DIR + 'test_df.pkl')

remove_features = pd.read_pickle(DIR + 'remove_features.pkl')
remove_features = list(remove_features['features_to_remove'].values)
print('Shape control:', train.shape, test.shape)

# load my feature
USE_FEATURE = ['f206', 'f103']
train, test = utils.load_feature(train, test, USE_FEATURE)

train = utils.reduce_mem_usage(train)
tset  = utils.reduce_mem_usage(test)

features_columns = [col for col in list(train) if col not in remove_features]

X_train = train[features_columns]
y_train = train['isFraud']
X_test  = test[features_columns]
y_test  = test['isFraud']

sub = test[['TransactionID', 'isFraud']]

trn_idx, val_idx = my_split(train)
trn_idx_list, oof_idx_list = ksplit(X_train.iloc[trn_idx], y_train.iloc[trn_idx])


# trn_idx_list, val_idx_list = ksplit(X_train, y_train)
del train
gc.collect()

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
val_x, val_y = X_train.iloc[val_idx], y_train.iloc[val_idx]  # validation fixing
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

    model.save_model(f'{save_model_dir}/fold{i}.pkl')

    val_preds = model.predict(val_x, num_iteration=model.best_iteration)
    oof_preds[oof_idx_list[i]] = model.predict(X_train.iloc[oof_idx_list[i]], num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_preds / NFOLD
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

    print(f'Fold {i+1} finished in {str(datetime.timedelta(seconds=time() - start_time))}\n')

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