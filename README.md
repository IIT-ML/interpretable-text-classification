# Interpretable and Cautious Text Classification



## Description
This is a collection of source code necessary for reproducibility.   <br>

## Dataset
There are two main data folders: dataset and data.
1. The dataset folder contains the datasets used in our experiments: 1) IMDB, 2) ArXiV, 3) AGnews.
    We're providing all of the datasets except for IMDB, which is publicly available, in our submission.
2. The data folder contains our generated lists of keywords for each dataset.


## Source Code
The root directory consists of two main file types: .py and .sh <br>
To reproduce the results, please run the command line as stated in the .sh files. <br>

### shell script
1. run_baseline.sh: to reproduce the results from Logistic Regression
2. run_hierarchical_attention.sh: to reproduce the results from HN and HAN (you need Glove Embedding)
3. run_cautious.sh: to reproduce the results from our model


### Python Code

1. train.py: Main .py script of our model
2. train_baseline.py: .py script for Logistic Regression
3. train_hierarchical.py: .py script for HN and HAN

## Parameter
Note that we're trying to uniform our model's setup linearly with Logistic Regression, thus the total number of parameter in our model is linear to the input document in initial assessment f_D + final classification f_C.


***Please follow the shell script
