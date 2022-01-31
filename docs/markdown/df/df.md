### df_add_scorecol


```python
df_add_scorecol(metric, win_threshold=0.66, lose_threshold=0.33, win_points=2, lose_points=-2, tie_points=0, score_col_name='score')
```


Adds an arbitrary score based on a metric column
and parameters that set the score boundaries. Takes
the win and lose thresholds, and applies them to
inter quantile range on the metric column. What is above
win threshold becomes a win, and what is below loss becomes
a lose, and everything in between becomes a tie.

data | DataFrame | a pandas dataframe with the data
metric | str | the column to be used for basis of scoring
win_threshold | float | the iqr at which point a win is consider
lose_threshold | float | the iqr at which point a lose is considered
win_points | int | the number of points for a win
lose_points | int | the number of points for a lose
tie_points | int | the number of points for a tie (between win and lose)
score_col_name | str | name to be used for the score column


----

### df_clean_colnames


```python
df_clean_colnames(destructive=False)
```



data : dataframe
A pandas dataframe with the data to be transformed
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.

TODO: (do we) need to add support for ordered/ordinal categoricals?


----

### df_corr_any


```python
df_corr_any(correlation='pearson')
```


Correlation for columns in a dataframe. Columns with
non-numeric values are ignored. Columns with numeric values
but wrong dtype are converted to numeric first.

data : pandas dataframe
The input data for the correlation
correlation : str
Method to be used for correlation: 'pearson', 'spearman', 'kendall'


----

### df_corr_extratrees


```python
df_corr_extratrees(y)
```


'Extra Trees Classifier based correlation.

data : pandas dataframe
The dataset to perform the correlation on
y : str
Column name for a categorical feature such as


----

### df_corr_ols


```python
df_corr_ols(y, destructive=False)
```


1-by-1 correlation with OLS for categorical values as strings.

Takes in a single column DataFrame where some columns are string
values of categories. Then performs an OLS test individually on
values of column returning an ordinal category representation (the greater
the value, the higher the correlation coefficient.)

data : dataframe
A Pandas dataframe with the data
y : str
The prediction variable.
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.


----

### df_corr_pearson


```python
df_corr_pearson(y, excluded_list=False)
```


One-by-One Correlation

Based on the input data, returns a dictionary where each column
is provided a correlation coefficient and number of unique values.
Columns with non-numeric values will have None as their coefficient.

data : dataframe
A Pandas dataframe with the data
y : str
The prediction variable
excluded_list : bool
If True, then also a list will be returned with column labels for
non-numeric columns.


----

### df_corr_randomforest


```python
df_corr_randomforest(y, destructive=True)
```


Random Forrest based correlation

Takes in a DataFrame with one or more columns of numeric
data in addition to 'y' which is continuous.

data : dataframe
A Pandas dataframe with the data
y : str
The prediction variable.
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.


----

### df_count_uniques


```python
df_count_uniques()
```


Returns the number of unique values in the columns of a dataframe
----

### df_count_uniques


```python
df_count_uniques()
```


Returns the number of unique values in the columns of a dataframe
----

### df_drop_col


```python
df_drop_col(cols, destructive=False)
```


Drops one or more columns from a dataframe
----

### df_drop_duplicates


```python
df_drop_duplicates()
```

----

### df_drop_nancols


```python
df_drop_nancols(threshold=0, destructive=False)
```


Drop columns based on a threshold of nan values. The optimal
approach, especially for small datasets, seems to be to first
start with df_drop_nancols, then follow with df_drop_nanrows,
and then do one more pass with df_drop_nancols.

data : pandas dataframe
A pandas dataframe to be processed
threshold : float
A decimal value between 1 (all can be nans) and 0 (no nans allowed).
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.


----

### df_drop_nanrows


```python
df_drop_nanrows(destructive=False)
```


Drop rows based on presences of nan values. The optimal
approach, especially for small datasets, seems to be to first
start with df_drop_nancols, then follow with df_drop_nanrows,
and then do one more pass with df_drop_nancols.

data : pandas dataframe
A pandas dataframe to be processed
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.


----

### df_drop_weak


```python
df_drop_weak(y, min_correlation=0.2, method='pearson', drop_object=True, destructive=False)
```


Keeps features that have a degree of correlatio with the prediction
feature. Drops other columns from the dataframe.

data : dataframe
A Pandas dataframe with the data
y : str
The prediction variable
min_correlation : float
The minimum absolute accepted correlation value
method : str
One of 'spearman', 'pearson', or 'kendall'
drop_object : bool
If object dtype columns should be dropped
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.


----

### df_fill_empty


```python
df_fill_empty(fill_with)
```


Finds and replaces any value in dataframe that only consist of
whitespace. A common scenario is where you first fill empties
with np.nan and then handle nans as you would otherwise do it.

NOTE: this results in all columns to be converted into strings. You can use
wrangle.df_to_numeric() to convert all back to numeric.

data : Pandas Dataframe
The dataframe to be processed
fill_with: str
Fill the values with a string.

----

### df_find_nan


```python
df_find_nan()
```



Finds nans and provides a dataframe with the column names and fraction of
nans in the column.


----

### df_groupby_params


```python
df_groupby_params(metric, dims=4, gram_type='ngram', group_by_func='sum')
```


Takes in a dataframe where each column contains
parameter labels, and then creates combinations of the
parameters based on ngram, skipgram, or flexgram approach.

data | DataFrame | contains the columns with parameters
dims | int | number of parameter dimensions in the output
gram_type | str | 'ngram', 'skipgram', 'flexgram', 'none'
group_by_func | str | one of the supported groupby functions


----

### df_impute_nan


```python
df_impute_nan(cols='all', impute_mode='mean_by_std', destructive=False)
```


Impute NaN values in a dataframe

Provides five different options for imputing nan
values within a a series / array of data.

data : DataFrame
A pandas dataframe with the data.
cols : 'all' or list
By default all columns will be imputed. Alternatively
accepts a list of columns as input.
impute_mode : str
The default is 'mean_by_std', with other
options 'mean', 'median', 'mode', and 'common'.
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.


----

### df_merge


```python
df_merge(data2, on_index=True, on_column=False)
```


Dataframe Merger

WHAT: Performs a pandas dataframe merge.

HOW: df_merge(data1, data2)

INPUT: two pandas dataframes

OUTPUT: a single pandas datafame


----

### df_parallelize_process


```python
df_parallelize_process(func, threads=16)
```


Parallelize data processing on a dataframe.

data | DataFrame | a pandas dataframe with the data
func | function | the function to be applied onto the input data
threads | int | number of threads to be used (max 2x your cores)

----

### df_print_values


```python
df_print_values(n=5)
```


Shows the most common values for each column
in a dataframe.

data | DataFrame | a pandas dataframe with the data
n | int | Number of most common values to show

----

### df_rename_col


```python
df_rename_col(col, rename_to, destructive=False)
```


Rename a single column

data : pandas DataFrame
Pandas dataframe with the column to be renamed.
col : str
Column to be renamed
rename_to : str
New name for the column to be renamed

destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.


----

### df_rename_cols


```python
df_rename_cols(exclude=None, prefix='C', destructive=False)
```


Generate sequential alphabetic names for columns
Takes in a dataframe and generates alphabetic
column names automatically.
data : pandas dataframe
A dataframe with one or more columns to be renamed.
exclude : str
A column, such as the prediction feature, to be excluded.
prefix : str
The prefix to be appended before the sequential number
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.

----

### df_resample_id


```python
df_resample_id(id_col, n=None, destructive=False)
```


Resamples a dataframe based on an id single column

Create a sample where each id has a single observation
in the whole dataset.

data : DataFrame
A pandas dataframe with at least one column
with an identifier (id_col) and a data column.
id_col : str
The column with the id for limiting uniqueness
n : int
The size of the sample to be drawn
destructive : bool


----

### df_resample_stratified


```python
df_resample_stratified(strat_col, n=None, destructive=False)
```


Stratified sampling of a DataFrame

data : pandas dataframe
A dataframe with one or more columns to be renamed.
strat_col : str
The column with categorical values that will be used
for stratification.
n : None or int
If int value, then that many samples will be present for each
class. Otherwise the number of samples for the label with the
lowest number of observations (samples) is used.
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.

----

### df_rescale_log


```python
df_rescale_log(retain_cols=None, destructive=False)
```


Transform dataframe to log1p.

data : pandas dataframe
A dataframe to be rescaled
retain_cols : str or list
The columns that should be excluded from rescaling
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.

----

### df_rescale_meanzero


```python
df_rescale_meanzero(retain=None)
```


ZERO MEAN SCALER
Takes in a dataframe or series, and rescales features
so that mean for each feature becomes 0 and standard
deviation becomes 1.
USE
===
mean_zero(df[['Age','Fare','Pclass']],retain='Pclass')
PARAMETERS
==========
data :: pandas dataframe or series object
retain :: if some column should be excluded from rescaling

----

### df_rescale_sqrt


```python
df_rescale_sqrt(retain_cols=None, destructive=False)
```



data : pandas dataframe
A dataframe to be rescaled
retain_cols : str or list
The columns that should be excluded from rescaling
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.

----

### df_restructure_values


```python
df_restructure_values(structure='list', destructive=False)
```


Takes in a dataframe, and restructures
the values so that the output dataframe consist
of columns where the value is coupled with the
column header.

data | DataFrame | a pandas dataframe
structure | str | 'list', 'str', 'tuple',  or 'dict'
destructive | bool | if False, the original dataframe will be retained


----

### df_to_binary


```python
df_to_binary(y, destructive=False)
```


USE col_to_binary instead.

----

### df_to_dfs


```python
df_to_dfs(groupers, y)
```


Converts a dataframe to dict of dataframes

Takes in a dataframe where a column name prefix
or similar can be used as a grouper for columns
and returns a dictionary with corresponding keys
for each subset dataframe.

USE: df_to_dfs(df, handler=['First', 'Last', 'Median'], 'expire_flag')

data : pandas DataFrame
A pandas dataframe with an outcome feature and more than one
columns with a label that have some common nominator in their
naming.
groupers : list
One or more groupers to be used as an identifier for the
split (the common nominator).
y : str
The outcome feature to be included in all the dataframes.


----

### df_to_groupby


```python
df_to_groupby(by, func)
```


Takes in a dataframe and returns it in a grouped by format.

data : dataframe
A pandas dataframe
by : str
The column by which the grouping is done
func : str
The function to be used for grouping by: 'median', 'mean', 'first',
'last', 'std', 'mode', 'max', 'min', 'sum', 'random', 'freq', 'string'.

----

### df_to_lower


```python
df_to_lower(cols=None)
```


Convert all string values to lowercase

data : pandas dataframe
The dataframe to be cleaned
cols : str, list, or None
If None, an attempt will be made to turn
all string columns into lowercase.


----

### df_to_multiclass


```python
df_to_multiclass(max_uniques=30, ignore_y=None, destructive=False)
```


Transforms string or other values to multi-label (1d) categorical.
Generally it might be a good idea to handle NaN values first, but in
case NaN values are present, they are converted into -1.

data : dataframe
A pandas dataframe with the data to be transformed
max_uniques : int
A threshold at which a column is no longer converted to a category
ignore_y : str
The string column name for y value
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.

TODO: (do we) need to add support for ordered/ordinal categoricals?
----

### df_to_multilabel


```python
df_to_multilabel(max_uniques=30, ignore_y=None, destructive=False)
```


Transforms string or other values to multi-class (n-d) categorical.
Generally it might be a good idea to handle NaN values first, but in
case NaN values are present, they are converted into to their own category.

EXAMPLE USE:
------------

to_multiclass(data)

PARAMS:
-------

data : dataframe
A pandas dataframe with the data to be transformed
max_uniques : int
A threshold at which a column is no longer converted to a category
ignore_y : str
The string column name for y value
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.
----

### df_to_numeric


```python
df_to_numeric(destructive=False)
```


Convert numeric values to ints and floats.

data : pandas dataframe
A pandas dataframe to be transformed

Takes in a dataframe and attempts to convert numeric columns
into floats or ints respectively.
----

### df_to_xy


```python
df_to_xy(y_col)
```


Split dataframe into x and y arrays

data : pandas dataframe
A pandas dataframe with the column to be converted
col : str
The column with the y values
