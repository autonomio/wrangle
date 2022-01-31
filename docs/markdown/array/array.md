### array_detect_task


```python
array_detect_task()
```


Detects the prediction task type based on an array
containing the prediction task labels.
----

### array_random_shuffle


```python
array_random_shuffle(y)
```


Randomize two arrays in same order
----

### array_random_weighted


```python
array_random_weighted(weights, size)
```


Weighted random pick from array

EXAMPLE:

elements = ['one', 'two', 'three']
weights = [0.2, 0.3, 0.5]
array_random_weighted(elements, weights)

x : array
Numpy array to draw the sample from
weights : list or 'normal'
Either a list of weights or 'normal' for
normal distribution random pick.
size : int

----

### array_reshape_conv1d


```python
array_reshape_conv1d()
```


Reshapes regulard 2d array for Conv1D layer

x : numpy array
A 2-d numpy array to be respahed.

----

### array_reshape_lstm


```python
array_reshape_lstm(seq_len, normalise_window)
```


Reshapes a 1d array for LSTM layer in Keras.

data : numpy array
A 2-d numpy array to be respahed.
seq_len : int
The length of the sequence of data
normalize_windown : int
The length of data normalization window

----

### array_split


```python
array_split(y, split, shuffled=True)
```


VALIDATION SPLIT OF X AND Y
Based on the Scan() parameter val_split
both 'x' and 'y' are split.

----

### array_to_generator


```python
array_to_generator(y, batch_size)
```


Creates a data generator for Keras fit_generator(). 
----

### array_to_kfold


```python
array_to_kfold(y, folds=10, shuffled=True)
```

----

### array_to_multilabel


```python
array_to_multilabel()
```


Converts a 1-d array with multiclass
integer labels into corresponding 2-d multilabel
version.