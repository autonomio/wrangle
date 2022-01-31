### col_check_allsame


```python
col_check_allsame(col)
```


Checks if all values in a column
have the same value. This can be detrimental
to a deep learning model.

data : DataFrame
col : str


----

### col_corr_category


```python
col_corr_category(col, outcome, rounding=2, warning_threshold=40)
```


Compute likeliness of a given
outcome against a set of class
attributes.

USE: wr.col_corr_cut(first, 'BUN_groups', 'hospital_mortality')

data : pandas DataFrame
col : str
A column with multilabel values
outcome : str
A column with binary class values
rounding : int
Number of decimals to be used for rounding.
warning_threshold : int
Number of samples required to avoid warning.
Warning is expressed as ** following the class
name.


----

### col_corr_ols


```python
col_corr_ols(x, y, destructive=False)
```


1-by-1 correlation with OLS for categorical values as strings.

Takes in a single column 'x' of string values representing
categories and performs an OLS test against a continuous
'y' column. Returns an ordinal category representation (the greater
the value, the higher the correlation coefficient.)

data : dataframe
A Pandas dataframe with the data
x : str
A feature with categorical/string values.
y : str
The truth variable. Has to be continuous.
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.

TODO: Mark low correlation coefficients labels all as 0.

----

### col_drop_outliers


```python
col_drop_outliers(col, mode='zscore', threshold=3, destructive=False)
```


Drop rows based on outliers in a given column. Note that
this drops NaN automatically.

data : pandas dataframe
A pandas dataframe to be processed
col : str
Name of the column to be processed. Note that rows will be
dropped for the whole dataframe.
mode : str
The mode to be used for outlier detection. Either 'zscore' or 'iqr'
threshold : int
The threshold to be used for IQR based outlier removal.


----

### col_fill_nan


```python
col_fill_nan(cols, fill_with=0)
```



Fills nans in a specific column or a range of
columns, using np.isnull() through pandas.

This is for the cases where some nans are dropped,
but some nans are converted in to zero (or other)
first.


----

### col_groupby_cdf


```python
col_groupby_cdf(col, metric_col, ascending=False)
```


Cumulative Distribution Function

Takes in a dataframe with at least 2 columns, and returns a
groupby table with CDF.

data | DataFrame | a pandas dataframe with the data
col | str | name of the column to be grouped by
metric_col | str | name of the column to be evaluated against
ascending | bool | the direction of sorting to be applied


----

### col_groupby_pdf


```python
col_groupby_pdf(col, metric_col, ascending=False)
```


Cumulative Distribution Function

Takes in a dataframe with at least 2 columns, and returns a
groupby table with PDF.

data | DataFrame | a pandas dataframe with the data
col | str | name of the column to be grouped by
metric_col | str | name of the column to be evaluated against
ascending | bool | the direction of sorting to be applied


----

### col_groupby_stats


```python
col_groupby_stats(col, y)
```


Takes in category column and returns
descriptive statistics based on an
binary outcome feature.

data :
col :
y :


----

### col_impute_nan


```python
col_impute_nan(impute_mode='mean_by_std')
```


Impute NaN values in a DataFrame column

Provides five different options for imputing nan
values within a a series / array of data.

USE: nan_imputer(titanic.Age,'mean')

OPTIONS: The default is 'mean_by_std', with other
options 'mean', 'median', 'mode', and 'common'.


----

### col_move_place


```python
col_move_place(col, position='first', destructive=False)
```


Moves column/s to being first or last column/s.

df : pandas dataframe
A pandas dataframe with the column/s to be moved
col : str or list
The column or columns to be moved
position : str
Either 'first' or 'last'
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.

----

### col_resample_equal


```python
col_resample_equal(col, sample_size)
```


Resamples based on unique labels in a column.
Results in a dataframe with equal number of rows
per label in a given column.

----

### col_resample_interval


```python
col_resample_interval(x, dt_col, mode='first', freq=60)
```


Resample a dataframe by creating intervals based on a datetime
column.

data : pandas dataframe
A dataframe with the values and a datetime column
x : str
The column where the actual data is
dt_col : str
The column with the datetime values
mode : str
The choices are 'median', 'mean', 'mode', 'first', 'last', 'std',
'mode', 'max', 'min', 'sum', 'random', 'freq'
freq : int
Number of minutes per sample as int or a string 'quarter', 'half',
'full' (days), 'week', 'month' (30 days), 'year'.

----

### col_rescale_max


```python
col_rescale_max(scale=1, to_int=False)
```


MinMax Rescaler

WHAT: rescales a set of values on to a fixed scale.

HOW: max_rescale([10,6,2],1)

INPUT: an array, list or Series

OUTPUT: an array with rescaled values.


----

### col_to_biclass


```python
col_to_biclass(col, true_value, destructive=False)
```


Takes in a dataframe column with two classes in
string form, and converts it to integer binary class.

data : pandas dataframe
A pandas dataframe with the column to be converted
col : str
The column with the multiclass values
true_value : str
The column value to be considered as True
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.

----

### col_to_binary


```python
col_to_binary(col, func='median', destructive=False)
```


Takes in a continuous feature and transforms into
a binary class.

df : pandas dataframe
A pandas dataframe with the column to be converted
col : str
The column with the multiclass values
func : str, float, or int
'mean','median','mode',int (ge), string for
interquartile range for binary conversion. 'cat_string'
for converting strings in to categorical labels, and
'cat_int' for doing the same with integer values.
destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.

----

### col_to_buckets


```python
col_to_buckets(col, cuts=5, rounding=False)
```


Cut continuous feature into equally sized buckets

USE: first['BUN_groups'] = test_cut(first, 'BUN', rounding=True)

data : pandas DataFrame
The dataframe with the data.
col : str
The column with thec continuous values.
cuts : int
Number of buckets to create.
rounding : bool
If True, then values will be rounded to zero decimal.


----

### col_to_cols


```python
col_to_cols(label_col, handler_col)
```


Convert a single multiclass column into columnsself.

data : pandas dataframe
A pandas dataframe to be transformed
label_col : str
A column with categorical labels (will become columns)
handler_col : str
A column with an identifier (e.g. country) which will become
an index column.

----

### col_to_multilabel


```python
col_to_multilabel(col, colnames=None, extended_colname=False, extended_separator='_')
```


Takes in a dataframe column with multiple labels and converts it
into several columns with binary labels.

df : pandas dataframe
A pandas dataframe with the column to be converted
col : str
The column with the multiclass values
colnames : list
A list of column names for the new values from 0 onwards
extended_colname : bool
If True, then the original column name will be a prefix for the
new column names.
extended_separator : str
The separator to be used in case `extended_colname` is in play.

----

### col_to_split


```python
col_to_split(col, col_names=None, sep=' ', destructive=False)
```


Splits a single column to multiple columns. Replaces
the original column so use destructive=True accordingly.

NOTE: each row has to have the same number of items after splitting.

data : pandas dataframe
The dataframe to be modified.
col : str
column to be split
col_names : list
Optional list of names of the new columns.
sep : str
Separator to split upon.

destructive : bool
If set to True, will make changes directly to the dataframe which
may be useful with very large dataframes instead of making a copy.
