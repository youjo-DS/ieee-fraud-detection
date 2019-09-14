import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def max_disp(df, rows=False, cols=False):
    if rows:
        pd.set_option('display.max_rows', None)
    if cols:
        pd.set_option('display.max_columns', None)
    display(df)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    
def get_info(target_df, topN=10, zero=False, u_val=False):
    max_row = len(target_df)
    print(f'Shape: {target_df.shape}')
    
    df = target_df.dtypes.to_frame()
    df.columns = ['DataType']
    df['Nulls'] = target_df.isnull().sum()
    df['Null%'] = df['Nulls'] / max_row * 100
    df['Uniques'] = target_df.nunique()
    df['Unique%'] = df['Uniques'] / max_row * 100
    
    if zero:
        df['Zeros'] = (train_genba == 0).sum()
        df['Zero%'] = df['Zeros'] / max_row
    
    # stats
    df['Min']   = target_df.min(numeric_only=True)
    df['Mean']  = target_df.mean(numeric_only=True)
    df['Max']   = target_df.max(numeric_only=True)
    df['Std']   = target_df.std(numeric_only=True)
    
    # top 10 values
    df[f'top{topN} val'] = 0
    df[f'top{topN} cnt'] = 0
    df[f'top{topN} raito'] = 0
    for c in df.index:
        vc = target_df[c].value_counts().head(topN)
        val = list(vc.index)
        cnt = list(vc.values)
        raito = list((vc.values / max_row).round(2))
        df.loc[c, f'top{topN} val'] = str(val)
        df.loc[c, f'top{topN} cnt'] = str(cnt)
        df.loc[c, f'top{topN} raito'] = str(raito)
        
    if u_val:
        df['u_val'] = [target_df[col].unique() for col in cols]
        
    return df

def cats_trvste(train, test, cats, rotation=0):
    for i, c in enumerate(cats):
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 2, 1)
        sns.countplot(train[c].sort_values().fillna('__NA__'))
        plt.xticks(rotation=rotation)

        plt.subplot(1, 2, 2)
        sns.countplot(test[c].sort_values().fillna('__NA__'))
        plt.xticks(rotation=rotation)

        plt.tight_layout()
        plt.show()
        
def cont_trvste(train, test, cont, bins=100):
    for i, c in enumerate(cont):
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 2, 1)
        plt.hist(train[c], bins=bins)
        plt.title(f'tr_{c}')

        plt.subplot(1, 2, 2)
        plt.hist(test[c], bins=bins)
        plt.title(f'te_{c}')

        plt.tight_layout()
        plt.show()
        
def category_diff(train, test, categorical_feature):
    diff_df = train[categorical_feature].nunique().to_frame()
    diff_df.columns = ['NUnique']
    diff_df['tr_vs_te'] = np.nan
    diff_df['te_vs_tr'] = np.nan
    for f in categorical_feature:
        diff_df.loc[f, 'tr_vs_te'] = str(set(train[f].unique()) - set(test[f].unique()))
        diff_df.loc[f, 'te_vs_tr'] = str(set(test[f].unique()) - set(train[f].unique()))
        
    return diff_df