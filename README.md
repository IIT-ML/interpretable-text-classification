# Interpretable and Cautious Text Classification



## Description
This is a collection of source code necessary for reproducibility.   <br>

## Dataset
There are two main data folders: data and dataset.
1. Dataset folder is the path to the dataset used in our experiment: 1) IMDB, 2) ArXiV, 3) AGnews
    We're providing the dataset, except the public dataset (IMDB), into our submission.
    The data is included in this folder
2. Data folder contains keywords parquet data provided in this folder


## Source Code
This main folder contains of two main file: .py and .sh <br>
To reproduce the result, please run the command line as stated in .sh file. <br>

### shell script
1. run_baseline.sh: to reproduce the result by Logistic Regression
2. run_hierarchical_attention.sh: to reproduce the result by HN and HAN (you need Glove Embedding)
3. run_cautious.sh: to reproduce the result by our model


### Python Code

1. train.py: Main .py script to our model
2. train_baseline.py: .py script for Logistic Regression
3. train_hierarchical.py: .py script for HN and HAN.

## Parameter
Note that we're trying to uniform our model's setup linearly with Logistic Regression, thus the total number of parameter in our model is linear to the input document in Initial assessment f_D + final classification f_C. 


***Please follow the shell script