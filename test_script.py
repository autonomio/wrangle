#!/usr/bin/env python
import wrangle as wr
import pandas as pd

# get some data first
df = pd.read_csv('https://raw.githubusercontent.com/autonomio/datasets/master/autonomio-datasets/programmatic_ad_fraud.csv')
df_cont_cat = wr.df_drop_nanrows(df[['bouncerate1', 'bouncerate2', 'category']])
df_cont_cont = wr.df_drop_nanrows(df[['bouncerate1', 'bouncerate2', 'score_median']])

# test all the attributes starting with df_
df = wr.df_clean_colnames(df)
_null = wr.df_corr_any(df, 'pearson')
df = wr.df_to_numeric(df)
_null = wr.df_corr_extratrees(df_cont_cat, 'category')
_null = wr.df_corr_ols(df[:50], 'score_median')
_null = wr.df_corr_pearson(df, 'score_median')
_null = wr.df_corr_randomforest(df_cont_cont, 'score_median')
_null = wr.df_count_uniques(df)
_null = wr.df_drop_col(df, 'score_median')
_null = wr.df_drop_duplicates(df)
_null = wr.df_drop_nancols(df)
_null = wr.df_drop_nanrows(df)
_null = wr.df_drop_weak(df, 'score_median')
_null = wr.df_find_nan(df)
impute_mode = ['mean', 'median', 'mode', 'common', 'mean_by_std']
for mode in impute_mode:
    wr.df_impute_nan(df.head(50), impute_mode=impute_mode)
_null = wr.df_merge(df.head(50), df.head(50))
_null = wr.df_rename_cols(df)
_null = wr.df_resample_stratified(df.head(50), 'adnetworks')
_null = wr.df_rescale_log(df_cont_cont)
_null = wr.df_rescale_log(df)
_null = wr.df_rescale_meanzero(df)
_null = wr.df_rescale_sqrt(df)
_null = wr.df_to_binary(df, 'score_median')
_null = wr.df_to_groupby(df, 'category', 'mean')
_null = wr.df_to_lower(df)
_null = wr.df_to_multiclass(df)
_null = wr.df_to_multilabel(df)
_null = wr.df_to_numeric(df)
