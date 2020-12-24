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

import csv
import pandas


import re


def show_explanations(preds, 
                      corpus, 
                      explanation_vector, 
                      keywordBank, 
                      true_label=None, 
                      notebook=True, 
                      verbose=True, 
                      return_obj=False,
                     model='our'):
    """
    Show the explanation given explanation_vector.
    We need to modify the explanation IF we use sparse model or our model
    
    # Arguments
        preds:
        explanation_vector:
        model: ['our', 'sparse']
        
    """
    
    if notebook:
        # Show to the notebook if displayed on jupyter notebook
        from IPython import display
        
        if isinstance(keywordBank, list):
            pass
        else:
            k = sorted(list(keywordBank.connotation.keys()))
        
        
        if return_obj:
            return ColoredWeightedDoc(' '.join(corpus), 
                                           k, keywordBank, 
                                           explanation_vector,
                                           model,
                                           binary = True)
        else:
            display.display(ColoredWeightedDoc(' '.join(corpus), 
                                           k, keywordBank, 
                                           explanation_vector, 
                                           model,
                                           binary = True))
        
            if verbose:
                if preds != -1:
                    print('-'*50)
                    if true_label:
                        print('True label : {}'.format(true_label))
                    print('This document predicted as {} because it has {} keyword justified as shown below:'.format(preds,
                                                                                                   np.sum(explanation_vector!=0)))

                    print()
                    for i,key in enumerate(explanation_vector):
                        if key != 0:
                            print('- {}'.format(k[i]))
                else:
                    if true_label:
                        print('True label : {}'.format(true_label))
                    print('This document does not has explanation. The model rejected...')

class ColoredWeightedDoc(object):
    """
    This class only supported for our model.
    We need to declare different class, of different option
    for sparse model.
    
    """
    def __init__(self, 
                 doc, 
                 keyword,
                 keywordBank,
                 explanation_vector, 
                 model,
                 token_pattern=r"(?u)\b\w\w+\b", 
                 binary = False):
        
        self.doc = doc
        self.keyword = keyword
        self.keywordBank = keywordBank
        self.explanation_vector = explanation_vector
        self.binary = binary
        self.tokenizer = re.compile(token_pattern)
        self.model = model
        
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
                    
                    
                    if not self.binary or (vocab_index not in seen_tokens):
                        # Need to fix this for sparse model
                        
                        if self.model == 'our':
                            # Opposing to the prediction
                            if (self.explanation_vector[vocab_index] == 0): 
                                html_rep = html_rep + "<font size = 1, color=grey> " + token + " </font>"
                            # Agreeing to the prediction
                            elif self.explanation_vector[vocab_index] != 0: 
                                if self.keywordBank.connotation[token.lower()] == 1:
                                    html_rep = html_rep + "<font size = 5, color=blue> " + token + " </font>"
                                else:
                                    html_rep = html_rep + "<font size = 5, color=red> " + token + " </font>"
                            else: # neutral word
                                html_rep = html_rep + "<font size = 1, color=grey> " + token + " </font>"
                                
                        elif self.model == 'sparse':
                            
                            if (self.explanation_vector[vocab_index] != 0):
                                # positive word
                                if (self.keywordBank.connotation[token.lower()] == 1):
                                    html_rep = html_rep + "<font size = 5, color=blue> " + token + " </font>"
                                # negative word
                                else:
                                    html_rep = html_rep + "<font size = 5, color=red> " + token + " </font>"
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

        self.html = html_rep
        return html_rep

def vectorize_keywords_docs(X_train_corpus, 
                            X_test_corpus, 
                            keywordBank=None, 
                            return_cv=False,
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
    from sklearn.feature_extraction.text import CountVectorizer
    
    # For simplicity of the model, we only extract feature with appearance of
    # 100 in the whole training corpus
    MIN_FREQ = 100
    
    doc_cv = CountVectorizer(min_df=MIN_FREQ,
                            token_pattern=token_pattern,
                            lowercase=True,
                            binary=True)
    
    X_train = {}
    X_test = {}
    
    X_train['docs'] = doc_cv.fit_transform([' '.join(text) for text in X_train_corpus])
    X_test['docs'] = doc_cv.transform([' '.join(text) for text in X_test_corpus])
    
    if keywordBank:
        key_cv = CountVectorizer(vocabulary=sorted(list(keywordBank.connotation.keys())),
                                 token_pattern=token_pattern,
                                 lowercase=True,
                                 binary=True)
        X_train['keys'] = key_cv.fit_transform([' '.join(text) for text in X_train_corpus])
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
        
    if return_cv:
        return X_train, X_test, doc_cv
    else:
        return X_train, X_test


def get_keyword(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as keys:
            keyword = []
            for k in keys:
                keyword.append(k.strip())
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)
