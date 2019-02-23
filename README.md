<h1 align="center">
  <br>
  <a href="http://autonom.io"><img src="https://raw.githubusercontent.com/autonomio/wrangle/master/logo.png" alt="Wrangle" width="350"></a>
  <br>
</h1>

<h3 align="center">Data preparation for deep learning</h3>

<p align="center">

  <a href="https://travis-ci.org/autonomio/wrangle">
    <img src="https://img.shields.io/travis/autonomio/wrangle/master.svg?style=for-the-badge&logo=appveyor" alt="Talos Travis">
  </a>

  <a href="https://coveralls.io/github/autonomio/wrangle">
    <img src="https://img.shields.io/coveralls/github/autonomio/wrangle.svg?style=for-the-badge&logo=appveyor" alt="Talos Coveralls">
  </a>

</p>

<p align="center">
  <a href="#Wrangle">Wrangle</a> •
  <a href="#Key-Features">Key Features</a> •
  <a href="#Install">Install</a> •
  <a href="#Support">Support</a> •
  <a href="https://github.com/autonomio/wrangle/issues">Issues</a> •
  <a href="#License">License</a> •
  <a href="https://github.com/autonomio/wrangle/archive/master.zip">Download</a>
</p>
<hr>
<p align="center">
Wrangle provides the building blocks for entirely avoiding redundant, easy-to-automate, data preparation tasks.
</p>

### Wrangle

TL;DR

Wrangle dramatically simplifies 95% of data preparation tasks involved in advanced deep learning practice and provides the required building blocks for near-future automated machine intelligence workflows. Wrangle is created to solve the problem of avoiding beneficial workflow steps due to complexity, cognitive overhead, and the anxiety that comes with it.

### Key Features

Because of the large number of functions, many of which are frequently used in common deep learning data preparation workflows, Wrangle is notably focused on namespace. All functions are named in a way where the name explains exactly what can be expected in terms of capability. Let's dissect a few as an example:

In `col_to_binary` *col* refers to what is being processed, in this case a column of a dataframe. *to* refers to the particular process, in this case a conversion. *binary* refers to the output. In this case a given column in a dataframe is converted into binary values. For example, a continuous column is converted to binary classes based on if the values are below or above mean value. Similarly `array_reshape_conv1d` can be understood as taking in an array, and reshaping it to conv1d layer required shape.

Wrangle key features include:

- Resampling
- Transformation
- Renaming
- Grouping
- Merging
- Correlations
- Reshaping
- Cleaning

Wrangle works on **Linux, Mac OSX**, and **Windows** systems.

### Install

Stable version:

#### `pip install wrangle`

Daily development version:

#### `pip install git+https://github.com/autonomio/wrangle.git@daily-dev`

### Support

If you want ask a **"how can I use Wrangle to..."** question, the right place is [StackOverflow](https://stackoverflow.com/questions/ask).

If you found a bug or want to suggest a feature, check the [issues](https://github.com/autonomio/wrangle/issues) or [create](https://github.com/autonomio/wrangle/issues/new/choose) a new issue.


### License

[MIT License](https://github.com/autonomio/wrangle/blob/master/LICENSE)
