"""
Dataset helper implementation

Methods:
    load_imdb
    load_amazon (soon)

Usage:
    >>> X_train_corpus, y_train, X_test_corpus, y_test = dataset_helper.load_imdb(IMDB_PATH, lower=True, tokenize=True)
    
    
Author: Anneke Hidayat, Mitchell Zhen, Mustafa Bilgic
"""

import os
import logging
import numpy as np
import glob
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

os.environ['TZ'] = 'America/Chicago'
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s')

def load_imdb(path, shuffle=True, random_state=42, lower=False, tokenize=True):
    '''
    Load IMDB data from the original folders from sentiment
    directory
    
    # Arguments
        path: path to IMDB data
        shuffle: 0 or 1
        random_state: Integer. 
            Use when shuffle=True
        tokenize: 0 or 1
            1 = Clean HTML tags and tokenize
    # Returns
        X_train_corpus, y_train: Train sets (text, label)
        X_test_corpus , y_test: Test sets (text, label)
    '''
    
    assert os.path.exists(path), "IMDB path does not exist"
    
    train_neg_files = glob.glob(path+"/train/neg/*.txt")
    train_pos_files = glob.glob(path+"/train/pos/*.txt")
    
    X_train_corpus = []
    y_train = []
    
    for tnf in train_neg_files:
        f = open(tnf, 'r', encoding="utf8")
        line = f.read()
        X_train_corpus.append(line)
        y_train.append(0)
        f.close()
    
    for tpf in train_pos_files:
        f = open(tpf, 'r', encoding="utf8")
        line = f.read()
        X_train_corpus.append(line)
        y_train.append(1)
        f.close()
    
    logging.info("Train Data loaded.")
    
    test_neg_files = glob.glob(path+"/test/neg/*.txt")
    test_pos_files = glob.glob(path+"/test/pos/*.txt")
    
    X_test_corpus = []
    y_test = []
    
    for tnf in test_neg_files:
        f = open(tnf, 'r', encoding="utf-8")
        X_test_corpus.append(BeautifulSoup(f.read().strip(), features="html5lib").get_text())
        y_test.append(0)
        f.close()
    
    for tpf in test_pos_files:
        f = open(tpf, 'r', encoding="utf-8")
        X_test_corpus.append(BeautifulSoup(f.read().strip(), features="html5lib").get_text())
        y_test.append(1)
        f.close()
    
    logging.info("Test Data loaded.")
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    
    X_train_corpus = [text.replace("<br />", "") for text in X_train_corpus]
    X_test_corpus = [text.replace("<br />", "") for text in X_test_corpus]
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_train))       
        
        X_train_corpus = [X_train_corpus[i] for i in indices]
        y_train = y_train[indices]
        
        indices = np.random.permutation(len(y_test))
        
        X_test_corpus = [X_test_corpus[i] for i in indices]
        y_test = y_test[indices]
        logging.info('Shuffled.')
    
    if lower:
        X_train_corpus = [text.lower() for text in X_train_corpus]
        X_test_corpus = [text.lower() for text in X_test_corpus]
        logging.info('Lowered.')
        
    if tokenize:
        X_train_corpus = [word_tokenize(text) for text in X_train_corpus]
        X_test_corpus = [word_tokenize(text) for text in X_test_corpus]
        logging.info('Tokenized.')
        
    return X_train_corpus, y_train, X_test_corpus , y_test

# TODO: load_amazon
def load_amazon(path, shuffle=True, 
                random_state=42, 
                lower=False, 
                tokenize=True, 
                train_test_split=True,
                test_split=None):

    def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

    def extract_review_amazon(path, key):
    corpus = []
    y = []
    text = parse(path)
    for l in text:
        corpus.append(l[key])
        y.append(l['overall'])
    return corpus, np.asarray(y)

    
    X, y = extract_review_amazon(path, 'reviewText')
    

        
