"""
Script to the model. 

Class:
    InterpretableCautiousText

Method:
    train
    test

Usage:
    >>> model = InterpretableCautiousText()
    
    
Author: Anneke Hidayat, Mitchell Zhen, Mustafa Bilgic

# TODO : separate model.py, test.py, train.py
"""

import os
# import tensorflow as tf
# import keras
import numpy as np
import json
import copy
import glob
import datetime
import pandas as pd
from pytz import timezone

from utils import utils, dataset_helper
from KeywordBank import KeywordBank
from nltk.tokenize import word_tokenize
from utils.dataset_helper import load_ag_news
RAND_SEED = 42

if __name__ == "__main__":
    import argparse

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Interpretable Cautious Text Classifier args")
    
    #### For temporary, the amazon_video and e_commerce won't be available
    parser.add_argument('--dataset', 
                        default='imdb',
                        help="dataset. {'imdb', 'arxiv', 'agnews'}. If path given, use the path")
    parser.add_argument('-t', '--testing', 
                        action='store_true',
                        help="test data with the given model path")
    parser.add_argument('--parent_dir', 
                        default='/home/anneke/Documents/models/', 
                        type=str,
                        help="Path to save model")
    
    
    args = parser.parse_args()
    # print(args)
    
    config = {}
    config['args'] = vars(args)
    config['start_time'] = datetime.datetime.now(timezone('US/Central')).strftime("%y-%m-%d_%H:%M:%S")
    
    #format(datetime.datetime.now(timezone('US/Central')).strftime("%y%m%d_%H%M%S"))
        
   
    # Load dataset
    if args.dataset.lower() == 'imdb':
        # 1. Load keyword from txt file
        # 2. Load dataset
        DATA_PATH = './dataset/aclImdb'
        KEYWORD_PATH = './data/imdb-keywords/imdb_keywords.json'
        
        print('{}: Load dataset'.format(config['start_time']))
        
        if os.path.exists(DATA_PATH) and os.path.exists(KEYWORD_PATH):
            
            #keyword = utils.get_keyword(KEYWORD_PATH)
            keyword = json.load(open(KEYWORD_PATH, 'r'))
            
            print('Loading...')
            X_train_corpus, y_train, X_test_corpus, y_test = dataset_helper.load_imdb(DATA_PATH, 
                                                                                      lower=True, 
                                                                                      tokenize=True)

            # 3. Create object to process keyword along with its connotation (keywordBank)
            keywordObj = KeywordBank(keyword=keyword, 
                                    xtrain=X_train_corpus, 
                                    ytrain=y_train)
            keywordObj.assign_connotation(words_len=args.word_len, 
                                          class_label=['neg', 'pos'])

            print('Vectorize...')
            # 4. Vectorize document and keyword for model input(s)
            X_train, X_test = utils.vectorize_keywords_docs(X_train_corpus, 
                                                            X_test_corpus, 
                                                            keywordObj)
        else:
            raise ValueError('Path doesn\'t exist. Please check the availability of your data')
    elif args.dataset.lower() == 'arxiv':
        DATA_PATH = '/home/anneke/Documents/ann-mitchell-text-classification/dataset/arxiv_cs_09_19_data.parquet'
        
        print('{}: Load dataset'.format(config['start_time']))
        
        if os.path.exists(DATA_PATH):
            print('Loading...')
            
            X_ = pd.read_parquet(DATA_PATH)
            
            
            categories = ['cs.lg',
                          'cs.cv',
                          'cs.it',
                          'cs.ai',
                          'cs.ds',
                          'cs.cl',
                          'cs.ni',
                          'cs.cr',
                          'cs.si',
                          'cs.dc']
            
            
        else:
            raise ValueError('Path doesn\'t exist. Please check the availability of your data')
    elif args.dataset.lower() == 'agnews':
        from itertools import combinations
        # Do combination to all categories
        
        DATA_PATH = '/home/anneke/Documents/ann-mitchell-text-classification/dataset/ag_news_csv/'
        
        print('{}: Load dataset'.format(config['start_time']))
        
        if os.path.exists(DATA_PATH):
            X_train, X_test, y_train, y_test = load_ag_news(DATA_PATH,
                                                            shuffle = True,
                                                            lower = True,
                                                            tokenize = True)
            
            X_train, X_test, cv = utils.vectorize_keywords_docs(X_train, 
                                                            X_test, return_cv=True)
            
            
            categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    else:
        pass
  

    ## Training here
    if args.dataset.lower() == 'agnews':
        
        directory = 'lr-{}-{}'.format(args.dataset, 
                                      config['start_time'])
        
        w_dir = 'weights/{}'.format(directory)
        
        if not os.path.exists(os.path.join(args.parent_dir, w_dir)):
            os.mkdir(os.path.join(args.parent_dir, w_dir))
            
            
            
        def get_categories(X, y, cat):
            '''
                    Make sure that the categories is always 2
            '''
                
            ind = np.array(list(np.where(y==cat[0])[0]) + list(np.where(y==cat[1])[0]))

            X_ = copy.deepcopy(X)
            y_cat = np.copy(y)

            X_['docs'] = X_['docs'][ind]
            y_cat = y_cat[ind]

            y_cat = [1 if y_==cat[1] else 0 for y_ in y_cat]
                
            return X_, np.array(y_cat)
        
            
        from sklearn.linear_model import LogisticRegression
        from itertools import combinations 
            
        comb = combinations([1, 2, 3, 4], 2)
        config['results'] = {}
        
        ### iter here
        for c in comb:
            config['results'][str(c)] = {}
            
            X_tr, y_tr = get_categories(X_train, y_train, c)
            X_te, y_te = get_categories(X_test, y_test, c)
            config['results'][str(c)]['categories'] = '({},{})'.format(categories[c[0]-1], categories[c[1]-1])
            config['results'][str(c)]['train_test_len'] = (len(y_tr), len(y_te))
            
                
            clf = LogisticRegression(penalty='l1', random_state=RAND_SEED)
            clf.fit(X_tr['docs'], y_tr)
            
            config['results'][str(c)]['train_acc'] = clf.score(X_tr['docs'], y_tr)
            config['results'][str(c)]['test_acc'] = clf.score(X_te['docs'], y_te)
            
            # Maybe get 50 top of words? From each category? 
            
            weight = clf.coef_[0]
            words = cv.get_feature_names()

            zero_indices = np.argsort(weight)
            one_indices = zero_indices[::-1]
            
            threshold = 50
            
            config['results'][str(c)]['{}-{}-words'.format(str(c[0]), threshold)] = []
            config['results'][str(c)]['{}-{}-words'.format(str(c[1]), threshold)] = []
                                      
            for i, (zero, one) in enumerate(zip(zero_indices[:threshold], one_indices[:threshold])):
                    config['results'][str(c)]['{}-{}-words'.format(str(c[0]), threshold)].append(words[zero])
                    config['results'][str(c)]['{}-{}-words'.format(str(c[1]), threshold)].append(words[one])
        
        config['end_time'] = datetime.datetime.now(timezone('US/Central')).strftime("%y-%m-%d_%H:%M:%S")

    elif args.dataset.lower() == 'arxiv':
        
        from sklearn.linear_model import LogisticRegression
        from itertools import combinations 
            
        comb = combinations(categories, 2)
        config['results'] = {}
        
        def apply_categories(data,
                             labels=['cs.ai', 'cs.cr']):
            '''
                Need to make sure that there is no overlap between these categories first!
            '''

            for l in labels:
                if l in data.split(' '):
                    return l

        ### iter here
        for c in comb:
            config['results'][str(c)] = {}
            
            X_['categories'] = X_['categories'].apply(apply_categories(labels))
            
            X_tr, y_tr = get_categories(X_train, y_train, c)
            X_te, y_te = get_categories(X_test, y_test, c)
            config['results'][str(c)]['categories'] = '({},{})'.format(categories[c[0]-1], categories[c[1]-1])
            config['results'][str(c)]['train_test_len'] = (len(y_tr), len(y_te))
            
                
            clf = LogisticRegression(penalty='l1', random_state=RAND_SEED)
            clf.fit(X_tr['docs'], y_tr)
            
            config['results'][str(c)]['train_acc'] = clf.score(X_tr['docs'], y_tr)
            config['results'][str(c)]['test_acc'] = clf.score(X_te['docs'], y_te)
            
            # Maybe get 50 top of words? From each category? 
            
            weight = clf.coef_[0]
            words = cv.get_feature_names()

            zero_indices = np.argsort(weight)
            one_indices = zero_indices[::-1]
            
            threshold = 50
            
            config['results'][str(c)]['{}-{}-words'.format(str(c[0]), threshold)] = []
            config['results'][str(c)]['{}-{}-words'.format(str(c[1]), threshold)] = []
                                      
            for i, (zero, one) in enumerate(zip(zero_indices[:threshold], one_indices[:threshold])):
                    config['results'][str(c)]['{}-{}-words'.format(str(c[0]), threshold)].append(words[zero])
                    config['results'][str(c)]['{}-{}-words'.format(str(c[1]), threshold)].append(words[one])
        
        config['end_time'] = datetime.datetime.now(timezone('US/Central')).strftime("%y-%m-%d_%H:%M:%S")
        
        
        X_['categories'] = X_['categories'].apply(apply_categories)
            
            
        from sklearn.model_selection import train_test_split
        print('train, test, split')
        X_train, X_test, y_train, y_test = train_test_split(X_['abstract'], 
                                                            X_['categories'], 
                                                            test_size=(1./3), 
                                                            random_state=42)
        del X_
        X_train = list(X_train)
        X_test = list(X_test)
            
        from nltk.tokenize import word_tokenize
        print('Tokenize...')
        X_train = [word_tokenize(text) for text in X_train]
        X_test = [word_tokenize(text) for text in X_test]
            
        to_binary = {'cs.ai':1, 'cs.cr':0}
        y_train = y_train.apply(lambda x: to_binary[x])
        y_test = y_test.apply(lambda x: to_binary[x])
        pass
    
    with open('{}/CONFIG'.format(os.path.join(args.parent_dir, w_dir)), 'w') as outfile:
            json.dump(config, outfile, indent=4)
           
    print('Finish...')