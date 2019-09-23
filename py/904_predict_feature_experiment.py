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

added_columns = [col for col in train.columns if not col in tr_cols_raw]

# =============================================================================
# usefull feature
# =============================================================================
useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',
                   'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                   'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M2', 'M3',
                   'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V17',
                   'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',
                   'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',
                   'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',
                   'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',
                   'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',
                   'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',
                   'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',
                   'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',
                   'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',
                   'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V261',
                   'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',
                   'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',
                   'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 'V323', 'V324', 'V326',
                   'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',
                   'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',
                   'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']

useful_features += added_columns

# =============================================================================
# preprocess
# =============================================================================
features = [col for col in test.columns if col in useful_features]

categorical_feature = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
                       'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
                       'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
categorical_feature = [col for col in categorical_feature if col in useful_features]
numeric_feature = [col for col in test.columns if col not in categorical_feature + ['TransactionID']]

# for c in categorical_feature:
#   if c in train.columns:
#     le = LabelEncoder()
#     le.fit(list(train[c].astype(str).values) + list(test[c].astype(str).values))
#     train[c] = le.transform(list(train[c].astype(str).values))
#     test[c]  = le.transform(list(test[c].astype(str).values))

for c in train.columns:
    if train[c].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(train[c].astype(str).values) + list(test[c].astype(str).values))
        train[c] = le.transform(list(train[c].astype(str).values))
        test[c]  = le.transform(list(test[c].astype(str).values))

if 'TransactionDT' in features:
    features.remove('TransactionDT')
if 'TransactionID' in features:
    features.remove('TransactionID')

print(f"""
using features:
{features}
""")

# columns = [col for col in test.columns if col not in 'TransactionID']
X_train = train[features]
y_train = train['isFraud']

test = test[features]

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

trn_idx_list, val_idx_list = ts_split(X_train, y_train)

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
    'num_leaves': 500,
    'min_data_in_leaf': 100,
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

# from kernel params
# params = {'num_leaves': 491,
#           'min_child_weight': 0.03454472573214212,
#           'feature_fraction': 0.3797454081646243,
#           'bagging_fraction': 0.4181193142567742,
#           'min_data_in_leaf': 106,
#           'objective': 'binary',
#           'max_depth': -1,
#           'learning_rate': 0.006883242363721497,
#           "boosting_type": "gbdt",
#           "bagging_seed": 11,
#           "metric": 'auc',
#           "verbosity": -1,
#           'reg_alpha': 0.3899927210061127,
#           'reg_lambda': 0.6485237330340494,
#           'early_stopping_round': 500,
#           'random_state': 47
#          }


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
    start_time = time()

    trn_x, trn_y = X_train.iloc[trn_idx_list[i]], y_train.iloc[trn_idx_list[i]]
    val_x, val_y = X_train.iloc[val_idx_list[i]], y_train.iloc[val_idx_list[i]]

    dtrain = lgb.Dataset(trn_x, label=trn_y, feature_name=features, categorical_feature=categorical_feature)
    dvalid = lgb.Dataset(val_x, label=val_y, feature_name=features, categorical_feature=categorical_feature)
    
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

    print(f'Fold {i+1} finished in {str(datetime.timedelta(seconds=time() - start_time))}')

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