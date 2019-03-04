#!/usr/bin/env python
import wrangle as wr
import pandas as pd
import numpy as np

# get some data first
df = pd.read_csv('https://raw.githubusercontent.com/autonomio/datasets/master/autonomio-datasets/programmatic_ad_fraud.csv')
df_cont_cat = wr.df_drop_nanrows(df[['bouncerate1', 'bouncerate2', 'category']])
df_cont_cat['binary'] = np.random.randint(0, 2, len(df_cont_cat))
df_cont_cont = wr.df_drop_nanrows(df[['bouncerate1', 'bouncerate2', 'score_median']])

# test all the attributes starting with df_
df = wr.df_clean_colnames(df)
_null = wr.df_corr_any(df, 'pearson')
df = wr.df_to_numeric(df)
_null = wr.df_corr_extratrees(df_cont_cat, 'category')
_null = wr.df_corr_ols(df[:500], 'score_median')
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

# test all the attributes starting with col_

_null = wr.col_check_allsame(df, 'category')
# _null = wr.col_corr_category(df, '')
_null = wr.col_corr_ols(df.head(50), 'bouncerate1', 'bouncerate1')
_null = wr.col_drop_outliers(df, 'bouncerate1', threshold=1)
_null = wr.col_fill_nan(df, 'admin_city')
_null = wr.col_impute_nan(df.bouncerate1)
_null = wr.col_move_place(df, 'bouncerate1', 'first')
_null = wr.col_move_place(df, 'bouncerate1', 'last')
_null = wr.col_resample_equal(df.head(50), 'adnetworks', 1)
_null = wr.col_rescale_max(df.bouncerate1.values)
_null = wr.col_to_binary(df, 'bouncerate1')
_null = wr.col_to_buckets(df, 'bouncerate1', 4)
_null = wr.col_to_cols(df[['adnetworks', 'bouncerate1']].reset_index(), 'adnetworks', 'index')
_null = wr.col_to_multilabel(df, 'category')
_null = wr.col_to_split(df.head(10), 'top_downstream', sep='.')

# test all the attributes starting with array_

_null = wr.array_random_shuffle(df.bouncerate1, df.bouncerate2)
_null = wr.array_random_weighted(df.bouncerate1.head(10), 'normal', 10)
_null = wr.array_reshape_conv1d(df.values)
_null = wr.array_reshape_lstm(df.bouncerate1, 10, 10)
_null = wr.array_split(df.values, df.bouncerate1.values, .1)
_null = wr.array_to_generator(df.values, df.bouncerate1, 20)
_null = wr.array_to_kfold(df.values, df.bouncerate1)
_nulll = wr.array_to_multilabel(df.head(5).adnetworks.values)

# test dict of dataframes (a 3d dataframe basically) attributes

_null = wr.dic_corr_perc(df, )


