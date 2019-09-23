import os
os.chdir('../py/')
import sys
import gc
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import feather
from glob import glob

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
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

# =============================================================================
# load feature
# =============================================================================
FEATURE_DIR = '../feature/'
USE_FEATURE = ['f103', 'f201']
if len(USE_FEATURE) > 0:
    tr_files = []
    te_files = []
    for f in USE_FEATURE:
        tr_file   = glob(f'../feature/{f}*tr*.ftr')
        tr_files += tr_file

        te_file   = glob(f'../feature/{f}*te*.ftr')
        te_files += te_file
else:
    tr_feature_path = '../feature/*tr*.ftr'
    te_feature_path = '../feature/*te*.ftr'

    tr_files = sorted(glob(tr_feature_path))
    te_files = sorted(glob(te_feature_path))

train = pd.concat([train, *[feather.read_dataframe(f) for f in tr_files]], axis=1)
test  = pd.concat([test, *[feather.read_dataframe(f) for f in te_files]], axis=1)


# =============================================================================
# preprocess
# =============================================================================
categorical_feature = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
                       'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
                       'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
numeric_feature = [col for col in test.columns if col not in categorical_feature + ['TransactionID']]

for c in categorical_feature:
  if c in train.columns:
    le = LabelEncoder()
    le.fit(list(train[c].astype(str).values) + list(test[c].astype(str).values))
    train[c] = le.transform(list(train[c].astype(str).values))
    test[c]  = le.transform(list(test[c].astype(str).values))

columns = [col for col in test.columns if col not in 'TransactionID']
X_train = train[columns]
y_train = train['isFraud']

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

trn_idx_list, val_idx_list = stratified(X_train, y_train)

# =============================================================================
# params
# =============================================================================

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': NROUND,
    'learning_rate': ETA,
    'max_depth': 16,
    'num_leaves': 500,
    # 'min_data_in_leaf': 80,
    # 'min_sum_hessian_in_leaf': 10,
    'min_child_weight': 3,
    'bagging_fraction': 0.9,
    'bagging_fraction': 0.9,
    'lambda_l1': 0.1,
    'lambda_l2': 0.01,
    'bagging_seed': SEED,
    'feature_fraction': 0.2,
    'early_stopping_round': 500,
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

    trn_x, trn_y = X_train.iloc[trn_idx_list[i]], y_train.iloc[trn_idx_list[i]]
    val_x, val_y = X_train.iloc[val_idx_list[i]], y_train.iloc[val_idx_list[i]]

    dtrain = lgb.Dataset(trn_x, label=trn_y, feature_name=columns, categorical_feature=categorical_feature)
    dvalid = lgb.Dataset(val_x, label=val_y, feature_name=columns, categorical_feature=categorical_feature)

    model = lgb.train(
        params,
        train_set=dtrain,
        valid_sets=[dtrain, dvalid],
        num_boost_round=NROUND,
        verbose_eval=500
    )

    val_preds = model.predict(val_x, num_iteration=model.best_iteration)
    oof_preds[val_idx_list[i]] = val_preds
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

mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(y_train, oof_preds)

print(f'Mean AUC: {mean_auc:.9f}, std: {std_auc:.9f}, All_AUC: {all_auc:.9f}')


# =============================================================================
# Save imp
# =============================================================================
importances.to_csv(f'LOG/imp/csv/imp_{__file__}.csv', index=False)

plt.figure(figsize=(16, int(len(imp_df) / 3)))
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