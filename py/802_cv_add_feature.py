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
NFOLD = 5
SEED  = 42
ETA   = 0.01

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
USE_FEATURE = ['f101']
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
# params
# =============================================================================

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'max_depth': -1,
    'num_leaves': 5,
#     'min_data_in_leaf': 80,
#     'min_sum_hessian_in_leaf': 10,
#     'bagging_fraction': 0.75,
#     'bagging_freq': 5,
#     'bagging_seed': SEED,
#     'feature_fraction': 0.2,
#     'metric': 'mape',
    'metric': 'auc',
#     'n_estimators': NROUND,
#     'early_stopping_round': 3000,
    'verbosity': -1,
#     'max_bin': 200,
#     'learning_rate': ETA
}


# =============================================================================
# cv
# =============================================================================

extraction_cb = custom_lgb.ModelExtractionCallback()
callbacks = [
    extraction_cb,
]

dtrain = lgb.Dataset(X_train, y_train.values, free_raw_data=False, categorical_feature=categorical_feature)

ret = lgb.cv(
  params=params,
  train_set=dtrain,
  num_boost_round=300,
  nfold=NFOLD,
  stratified=True,
  shuffle=True,
  early_stopping_rounds=30,
  verbose_eval=50,
  seed=SEED,
  callbacks=callbacks
)

# =============================================================================
# prediction and plot
# =============================================================================

boosters = extraction_cb.raw_boosters
best_iteration = extraction_cb.best_iteration

importances = pd.DataFrame()
test_preds = np.zeros(len(test))

for i, booster in enumerate(boosters):
  # prediction
  test_preds += booster.predict(test, num_iteration=best_iteration) / NFOLD

  # get importance
  imp_df = pd.DataFrame(
    {'feature': X_train.columns,
     'gain': booster.feature_importance(),
     'Fold': i+1}
  )
  imp_df = imp_df.head(100)
  importances = pd.concat([importances, imp_df], axis=0)


# =============================================================================
# Save imp
# =============================================================================
importances.to_csv(f'LOG/imp/csv/imp_{__file__}.csv', index=False)

plt.figure(figsize=(16, int(len(imp_df) / 3)))
sns.barplot(x='gain', y='feature', data=importances.sort_values('gain', ascending=False))
plt.savefig(f'LOG/imp/PNG/imp_{__file__}.png')

utils.end(__file__)