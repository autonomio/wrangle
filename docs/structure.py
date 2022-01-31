
from wrangle import array,col,df,dic,utils



EXCLUDE = {
 # put the modules to exclude here.
}

# For each class to document, it is possible to:
# 1) Document only the class: [classA, classB, ...]
# 2) Document all its methods: [classA, (classB, "*")]
# 3) Choose which methods to document (methods listed as strings):
# [classA, (classB, ["method1", "method2", ...]), ...]
# 4) Choose which methods to document (methods listed as qualified names):
# [classA, (classB, [module.classB.method1, module.classB.method2, ...]), ...]
PAGES = [
    {
        'page': 'array/array.md',
        
        'classes': [
          
        ],
        'methods': [
            array.array_detect_task,
            array.array_random_shuffle,
            array.array_random_weighted,
            array.array_reshape_conv1d,
            array.array_reshape_lstm,
            array.array_split,
            array.array_to_generator,
            array.array_to_kfold,
            array.array_to_multilabel
       
        ],
       
    },
       
 {
    'page': 'col/col.md',
    
            'classes': [
                      
       
    ],
    'methods': [
           col.col_check_allsame,
           col.col_corr_category,
           col.col_corr_ols,
           col.col_drop_outliers,
           col.col_fill_nan,
           col.col_groupby_cdf,
           col.col_groupby_pdf,
           col.col_groupby_stats,
           col.col_impute_nan,
           col.col_move_place,
           col.col_resample_equal,
           col.col_resample_interval,
           col.col_rescale_max,
           col.col_to_biclass,
           col.col_to_binary,
           col.col_to_buckets,
           col.col_to_cols,
           col.col_to_multilabel,
           col.col_to_split
    ],
     
},
 
 {
    'page': 'df/df.md',
    
            'classes': [
                      
       
    ],
    'methods': [
        df.df_add_scorecol,
        df.df_clean_colnames,
        df.df_corr_any,
        df.df_corr_extratrees,
        df.df_corr_ols,
        df.df_corr_pearson,
        df.df_corr_randomforest,
        df.df_count_uniques,
        df.df_count_uniques,
        df.df_drop_col,
        df.df_drop_duplicates,
        df.df_drop_nancols,
        df.df_drop_nanrows,
        df.df_drop_weak,
        df.df_fill_empty,
        df.df_find_nan,
        df.df_groupby_params,
        df.df_impute_nan,
        df.df_merge,
        df.df_parallelize_process,
        df.df_print_values,
        df.df_rename_col,
        df.df_rename_cols,
        df.df_resample_id,
        df.df_resample_stratified,
        df.df_rescale_log,
        df.df_rescale_meanzero,
        df.df_rescale_sqrt,
        df.df_restructure_values,
        df.df_to_binary,
        df.df_to_dfs,
        df.df_to_groupby,
        df.df_to_lower,
        df.df_to_multiclass,
        df.df_to_multilabel,
        df.df_to_numeric,
        df.df_to_xy
    ],
     
},
 {
     'page': 'dic/dic.md',
     
     'classes': [
       
     ],
     'methods': [
         dic.dic_corr_perc,
         dic.dic_count_complexity,
         dic.dic_resample_values,
    
     ],
    
 },
 {
     'page': 'utils/utils.md',
     
     'classes': [
       
     ],
     'methods': [
       utils.create_synth_binary_model,
       utils.create_synth_data,
       utils.create_synth_multi_class_model,
       utils.create_synth_multi_label_model,
       utils.create_synth_regression_model,
       utils.create_time_sequence,
       utils.read_large_csv
     ],
    
 },
]

ROOT = 'http://autonom.io/'


