"""
Utility helper

Methods:
    vectorize_keywords_docs

Usage:
    # reference on imdbKeyword see KeywordBank.py
    X_train, X_test = vectorize_keywords_docs(X_train_corpus, X_test_corpus, imdbKeyword)
    
    
Author: Anneke Hidayat, Mitchell Zhen, Mustafa Bilgic
"""

import os
import errno
import numpy as np
from matplotlib import pyplot as plt
import csv
import pandas

plt.style.use("seaborn")

import re


def show_explanations(preds, corpus, explanation_vector, keywordBank, notebook=True, verbose=True):
    """
    Show the explanation given explanation_vector
    
    # Arguments
        preds:
        explanation_vector:
        
    """
    
    if notebook:
        # Show to the notebook if displayed on jupyter notebook
        from IPython import display
        
        
        print('Document:')
        display.display(ColoredWeightedDoc(' '.join(corpus), 
                                           keywordBank.keyword, 
                                           explanation_vector, 
                                           binary = True))
        
        if verbose:
            if preds != -1:
                print('-'*50)
                print('This document predicted as {} because it has {} keyword justified as shown below:'.format(preds,
                                                                                               np.sum(explanation_vector!=0)))
            
                print()
                for i,key in enumerate(explanation_vector):
                    if key != 0:
                        print('- {}'.format(keywordBank.keyword[i]))
            else:
                print('This document does not has explanation. The model rejected...')

class ColoredWeightedDoc(object):
    """
    """
    def __init__(self, 
                 doc, 
                 keyword, 
                 explanation_vector, 
                 token_pattern=r"(?u)\b\w\w+\b", binary = False):
        self.doc = doc
        self.keyword = keyword
        self.explanation_vector = explanation_vector
        self.binary = binary
        self.tokenizer = re.compile(token_pattern)
        
    def _repr_html_(self):
        html_rep = ""
        tokens = self.doc.split(" ") 
        if self.binary:
            seen_tokens = set()       
        for token in tokens:
            vocab_tokens = self.tokenizer.findall(token.lower())
            if len(vocab_tokens) > 0:
                vocab_token = vocab_tokens[0]
                try:
                    vocab_index = self.keyword.index(vocab_token)
                    
                    if not self.binary or vocab_index not in seen_tokens:
                        
                        if self.explanation_vector[vocab_index] == 0: # Opposing to the prediction
                            html_rep = html_rep + "<font size = 4, color=orange> " + token + " </font>"
                        
                        elif self.explanation_vector[vocab_index] != 0: # Agreeing to the prediction
                            html_rep = html_rep + "<font size = 5, color=blue> " + token + " </font>"
                        
                        else: # neutral word
                            html_rep = html_rep + "<font size = 1, color=grey> " + token + " </font>"
                        
                        if self.binary:    
                            seen_tokens.add(vocab_index)
                    
                    else: # if binary and this is a token we have seen before
                        html_rep = html_rep + "<font size = 1, color=grey> " + token + " </font>"
                except: # this token does not exist in the vocabulary
                    html_rep = html_rep + "<font size = 1, color=grey> " + token + " </font>"
            else:
                html_rep = html_rep + "<font size = 1, color=grey> " + token + " </font>"
        return html_rep

def vectorize_keywords_docs(X_train_corpus, 
                            X_test_corpus, 
                            keywordBank, 
                            token_pattern=r"(?u)\b[\w\'/]+\b"):
    """ Vectorize the document input and the keyword input
    with respect to the keyword connotation
    
    # Arguments
        X_train_corpus: List
        X_test_corpus: List
        keywordBank: Object
        token_pattern: default (r"(?u)\b[\w\'/]+\b")
        
    # Returns
        doc_vec: document binary bag-of-words
        key_vec: keyword bag-of words with connotation
    """
    import sklearn
    
    # For simplicity of the model, we only extract feature with appearance of
    # 100 in the whole training corpus
    MIN_FREQ = 100
    
    doc_cv = sklearn.feature_extraction.text.CountVectorizer(min_df=MIN_FREQ,
                                                            token_pattern=token_pattern,
                                                            lowercase=True,
                                                            binary=True)
    key_cv = sklearn.feature_extraction.text.CountVectorizer(vocabulary=keywordBank.keyword,
                                                            token_pattern=token_pattern,
                                                            lowercase=True,
                                                            binary=True)
    
    X_train = {}
    X_test = {}
    
    X_train['docs'] = doc_cv.fit_transform([' '.join(text) for text in X_train_corpus])
    X_train['keys'] = key_cv.fit_transform([' '.join(text) for text in X_train_corpus])
    X_test['docs'] = doc_cv.transform([' '.join(text) for text in X_test_corpus])
    X_test['keys'] = key_cv.transform([' '.join(text) for text in X_test_corpus])
    
    # 1. Reverse the vocabulary lookup
    # 2. Add connotation for keys
    # CountVectorizer return {0,1} value.
    # For each {1} in the element, we replace with the connotation value
    key_index_lookup = dict([[v,k] for k,v in key_cv.vocabulary_.items()])
            
    tr_row_indices, tr_feat_indices = X_train['keys'].nonzero()
    te_row_indices, te_feat_indices = X_test['keys'].nonzero()
    
    # Since train and test could have different size of samples
    # We iterate the data separately
    for row, feat in zip(tr_row_indices, tr_feat_indices):
        X_train['keys'][row, feat] = keywordBank.connotation[key_index_lookup[feat]]
    for row, feat in zip(te_row_indices, te_feat_indices):
        X_test['keys'][row, feat] = keywordBank.connotation[key_index_lookup[feat]]
        
    return X_train, X_test


def plot_log(filename, show=True):
    # Taken from https://github.com/XifengGuo/CapsNet-Keras/blob/master/utils.py
    
    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()

def get_keyword(filepath):
    if os.path.exist(filepath):
        with open(filepath, 'r', encoding='utf-8') as keys:
            keyword = []
            for k in keys:
                keyword.append(k.strip())
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)