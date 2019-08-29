'''
This script contains baseline model to compare our model with
Logistic Regression would not be included here.

Hierarchical Network
Hierarchical Attention Network
'''

import numpy as np

from nltk import tokenize
from textblob import TextBlob


MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# sequence pre-processing
