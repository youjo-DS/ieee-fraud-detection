import pandas as pd
import numpy as np

from collections import OrderedDict

# 連続値から代表値を取得する
def auto_agg(data: pd.DataFrame, group, agg_cols, agg_funcs, orig=None, prefix='', suffix='', how='left', drop=False):
    """
    諸々を指定すると、集計・カラムのリネームを行い、最終結合用のDataFrameを返す。
    -------
    orig: DataFrame. 最後に戻すテーブルを設定する。（行数をそろえるため）
    data: DataFrame. 集計を行いたいDataFrameを設定する。
    group: list or str. 集計を行いたいカラムを設定する。
    agg_cols: list or list. 代表値を取得したいカラムを設定する。
    agg_func: list or list. 集約関数を設定する。
    prefix: str. 頭につけたい名前があれば指定する。
    suffix: str. 後ろにつけたい名前があれば指定する。
    how: str. マージ方法の指定。pandas の merge に依存。
    """
    if isinstance(agg_cols, str):
        agg_cols = [agg_cols]

    if isinstance(agg_funcs, str):
        agg_funcs = [agg_funcs]
   
    # dic = {col:agg_funcs for col in agg_cols} # make aggregation dictionary
    dic = OrderedDict(([[col, agg_funcs] for col in agg_cols]))
    agg_df = data.groupby(group).agg(dic)

    # make rename column list    
    col_names = [f'{prefix}{key}_{value}{suffix}' for key in dic for value in dic[key]]                
    agg_df.columns = col_names
    
    if orig is None:
        return agg_df

    # for final merge
    agg_df.reset_index(inplace=True)

    df = orig[group] # make dataset for merge
    del orig
    
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    
    df = df.merge(agg_df, on=group, how=how)

    if drop:
        df.drop(columns=group, inplace=True)

    return df

# カテゴリ値から個数と割合を算出する。
def auto_cnt_ratio(data: pd.DataFrame, group, tgt_cols, orig=None, prefix='', suffix='', how='left'):
    """
    諸々を指定すると、カテゴリの個数、割合の計算・カラムのリネームを行い、
    最終結合前のDataFrameを返す。
    ------
    orig: DataFrame. 最後に戻すテーブルを設定する。（行数をそろえるため）
    data: DataFrame. 集計を行いたいDataFrameを設定する。
    group: list or str. 集計を行いたいカラムを設定する。
    tgt_cols: list or str. カウントを取得したいカラムを設定する。
    prefix: str. 頭につけたい名前があれば指定する。
    suffix: str. 後ろにつけたい名前があれば指定する。
    how: マージ方法の指定。
    """
    if isinstance(group, str):
        group = [group]

    if isinstance(tgt_cols, str):
        tgt_cols = [tgt_cols]

    all_cols = group + tgt_cols
    df = data[all_cols] # to dataset smaller

    cnt_arr = []
    ratio_arr = []
    for tgt in tgt_cols:
        cnt_tmp_df = pd.crosstab(index=[df[f'{g}'] for g in group],
                                 columns=df[tgt])
        ratio_tmp_df = pd.crosstab(index=[df[f'{g}'] for g in group],
                                   columns=df[tgt],
                                   normalize='index')
        
        cnt_tmp_df.columns = [f'{prefix}{tgt}Cnt_{cat}{suffix}' for cat in list(cnt_tmp_df.columns)]
        ratio_tmp_df.columns = [f'{prefix}{tgt}Ratio_{cat}{suffix}' for cat in list(ratio_tmp_df.columns)]

        cnt_arr.append(cnt_tmp_df)
        ratio_arr.append(ratio_tmp_df)

    cnt_conc_df = pd.concat(cnt_arr, axis=1)
    ratio_conc_df = pd.concat(ratio_arr, axis=1)

    if orig is None:
        return cnt_conc_df, ratio_conc_df

    cnt_conc_df.reset_index(inplace=True)
    ratio_conc_df.reset_index(inplace=True)
    
    # make dataset for merge
    cnt_df = orig[group]
    ratio_df = orig[group]
    del data, orig

    cnt_df = cnt_df.merge(cnt_conc_df, on=group, how=how)
    ratio_df = ratio_df.merge(ratio_conc_df, on=group, how=how)
    
    return cnt_df, ratio_df

def merge_all(orig: pd.DataFrame, tables: list, group):
    """
    配列の長さが一致、結合キーが完全一致しているテーブルについて、
    merge ではなく、numpy array の concat で DataFrame を再生成する。
    -----
    orig: DataFrame. 最後に戻すテーブルを設定する。
    tables: list. 結合したいテーブルをリストで指定する。
    group: list or str. 集計を行いたいカラムを設定する。
    """
    assert type(tables) == list, 'tables must be list.'

    df = pd.concat([orig, *[t.drop(columns=group) for t in tables]], axis=1)

    return df
