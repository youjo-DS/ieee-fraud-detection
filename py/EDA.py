import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from copy import copy

# ==========================================================================
# display
# ==========================================================================
def max_disp(df, rows=False, cols=True):
    if rows:
        pd.set_option('display.max_rows', None)
    if cols:
        pd.set_option('display.max_columns', None)
    display(df)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

# ==========================================================================
# dataframe
# ==========================================================================
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

def category_diff(train, test, categorical_features):
    diff_df = train[categorical_features].nunique().to_frame()
    diff_df.columns = ['nUnique']
    diff_df['tr_vs_te'] = np.nan
    diff_df['te_vs_tr'] = np.nan
    for f in categorical_features:
        diff_df.loc[f, 'tr_vs_te'] = str(set(train[f].unique()) - set(test[f].unique()))
        diff_df.loc[f, 'te_vs_tr'] = str(set(test[f].unique()) - set(train[f].unique()))
        
    return diff_df

# ==========================================================================
# plot
# ==========================================================================
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

def continuous_vs_target(train, test, continuous_features, target, bins=100, alpha=0.8, fillna=False, edgecolor1='royalblue', edgecolor2='orange'):
    for i, c in enumerate(continuous_features):
        print(c)

        plt.figure(figsize=(16, 5))
        plt.subplot(1, 2, 1)
        plt.hist(train[train[target] == 0][c], bins=bins, alpha=alpha, edgecolor=edgecolor1, label='t = 0')
        plt.hist(train[train[target] == 1][c], bins=bins, alpha=alpha, edgecolor=edgecolor2, label='t = 1')
        plt.legend()
        plt.title('train')

        if test is not None:
            plt.subplot(1, 2, 2)
            plt.hist(test[c], bins=bins, edgecolor=edgecolor1)
            plt.title('test')

        plt.tight_layout()
        plt.show()
    return

def count_plot(train, categorical_features, test=None, topN=30, sort='freq', rotation=0, fillna=False, figsize=(16, 5)):
    
    tr = copy(train) 
    te = copy(test)
    
    if not isinstance(categorical_features, list):
        categorical_features = list(categorical_features)
    
    for c in categorical_features:
        print(c)
        if fillna:
            tr[c].fillna('__NA__', inplace=True)
            te[c].fillna('__NA__', inplace=True)

        traget_value = tr[c].value_counts().head(topN).index
        if sort == 'freq':
            order = traget_value
        elif sort == 'alphabetic':
            order = tr[c].value_counts().head(topN).sort_index().index
        
        if test is not None:
            plt.figure(figsize=figsize)
            plt.subplot(1, 2, 1)
        sns.countplot(x=c, data=train[tr[c].isin(order)], order=order)
        plt.xticks(rotation=rotation)

        if test is not None:
            plt.subplot(1, 2, 2)
            sns.countplot(x=c, data=test[test[c].isin(order)], order=order)
            plt.xticks(rotation=rotation)

        if test is not None:
            plt.suptitle(f'{c} TOP{topN}')
        plt.tight_layout()
        plt.show()
    return

def category_vs_target(train, test, categorical_features, target_column, sort='alphabetic', rotation=0, fillna=False, figsize=(16, 5)):
    tr = copy(train)
    te = copy(test)
    
    if not isinstance(categorical_features, list):
        categorical_features = list(categorical_features)
        
    for c in categorical_features:
        print(c)
        if fillna:
            tr[c].fillna('__NA__', inplace=True)
            te[c].fillna('__NA__', inplace=True)
            
        # count, target_mean calculation
        train_agg = tr.groupby(c, as_index=False).agg({f'{target_column}': ['mean', 'count']})
        train_agg.columns = [c, 'mean', 'count']
        train_agg['freq'] = train_agg['count'] / len(train)
        
        test_freq = te[c].value_counts(normalize=True).to_frame().reset_index()
        test_freq.columns = [c, 'freq']
        
        # sort rule
        if sort == 'alphabetic':
            train_agg.sort_values(c, inplace=True)
            test_freq.sort_values(c, inplace=True)
            train_order = train_agg[c].values
            test_order  = test_freq[c].values
            
        elif sort == 'freq':
            train_agg.sort_values('freq', ascending=False, inplace=True)
            test_freq.sort_values('freq', ascending=False, inplace=True)
            train_order = train_agg[c].values
            test_order  = test_freq[c].values
        
        # plot
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = ax1.twinx()

        ax1.bar(train_order, train_agg['freq'], label='frequency')
        ax1.set_ylim(0, 1)
        ax1.legend()
        labels1 = ax1.get_xticklabels()
        plt.setp(labels1, rotation=rotation)

        ax2.plot(train_order, train_agg['mean'], c='r', ls='--', marker='o', label='target_mean')
        ax2.grid(False)
        ax2.set_ylim(0, 1)
        ax2.legend(bbox_to_anchor=(1, 0.9))

        ax3 = fig.add_subplot(1, 2, 2)
        ax3.bar(test_order, test_freq['freq'])
        ax3.set_ylim(0, 1)
        labels2 = ax3.get_xticklabels()
        plt.setp(labels2, rotation=rotation)

        plt.tight_layout()
        plt.show()
        
    return

def category_target_percentile(train, categorical_features, target_column, percentile=True, stacked=True, figsize=None, fillna=False):
    if fillna:
        train = copy(train)
        train[categorical_features].fillna('__NA__', inplace=True)

    if percentile:
        normalize = 'index'
    else:
        norimalize = True

    for c in categorical_feature:
        pd.crosstab(
            index=train[c],
            columns=target_column,
            normalize=normalize
        ).plot(kind='bar', stacked=True, figsize=figsize)
        plt.show()

    return

def kde_train_vs_target(train, columns, target):
    if isinstance(columns, str):
        columns = [columns]
        
    t0 = train[train[target] == 0]
    t1 = train[train[target] == 1]
    for c in columns:
        print(c)
        plt.figure(figsize=(16, 5))
        sns.kdeplot(t0[c], shade=True)
        sns.kdeplot(t1[c], shade=True)
        plt.show()
        
    return