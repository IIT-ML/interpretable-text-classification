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
import pandas as pd
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

def load_ag_news(abs_path, 
                 tokenize=True,
                 lower=True,
                 shuffle=True,
                 random_state=42):
    """
    Load AG news with four categories
    """
    
    train_path = os.path.join(abs_path, 'train.csv')
    test_path = os.path.join(abs_path, 'test.csv')
    
    assert os.path.exists(abs_path), "AGnews path doesn\'t exists..."
    
    agnews_train_pd = pd.read_csv(train_path, header=None)
    agnews_test_pd = pd.read_csv(test_path, header=None)
    
    X_train_corpus = agnews_train_pd[2].tolist()
    X_train_corpus = [x.replace('\\', ' ') for x in X_train_corpus]
    y_train = np.array(agnews_train_pd[0].values)
 
    X_test_corpus = agnews_test_pd[2].tolist()
    X_test_corpus = [x.replace('\\', ' ') for x in X_test_corpus]
    y_test = np.array(agnews_test_pd[0].values)
    
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
    
    return X_train_corpus, X_test_corpus, y_train , y_test

def load_peer_read(abs_path,
                   tokenize=True, 
                   lower=True, 
                   shuffle=True, 
                   random_state=42):
    """
    Load data for peerRead
    
    # Arguments
        tokenize: Boolean
        lower: Boolean
        shuffle: Boolean
        random_state: Integer
    # Returns
        X_train, X_test, y_train, y_test
    
    """
    
    import json
    
    def get_parsed(path, 
                   data_mode, 
                   label_mode):
        """
        get the file names under different conference folder
        
        # Arguments
            path: Path to the parent folder
            data_mode: dict.
                See below on data_mode (key,value) -> ('train', 'path to train')
            label_mode: dict.
                same as data. But on label to get reviews
        
        """
        parsed_data = {}
        parsed_label = {}

        for key,item in data_mode.items():
            parsed_data[key] = sorted(glob.glob1(os.path.join(path, item),
                                         '*'))
        
        for key,item in label_mode.items():
            parsed_label[key] = sorted(glob.glob1(os.path.join(path, item),
                                          '*'))
            
        meta = [(key, len(item)) for key,item in parsed_data.items()]

        return meta, parsed_data, parsed_label
    
    def get_data(meta, data, label, data_mode, label_mode, mode=None):
        """
        
        # Args
            mode: {'train', 'test', 'dev'}
        # returns
            x, y
        
        """
        ignore = ['acl', 'iclr', 'coNLL']
        if mode is not None:
            x = []
            y = []
            for k in data.keys():
                for i in range(len(data[k][mode])):
                        # data
                        with open(os.path.join(path[k], data_mode[mode], data[k][mode][i])) as json_file:  
                            file = json.load(json_file) 
                            if file['metadata']['sections'] is not None:
                                temp = ' '
                                for key in file['metadata']['sections']:
                                    if key['heading'] is not None:
                                        temp = ' '.join([temp, '{} {}'.format(key['heading'], key['text'].strip())])
                        x.append(temp.replace('\n', ' '))

                        if k not in ignore:
                            # label
                            with open(os.path.join(path[k], label_mode[mode], label[k][mode][i])) as json_file:  
                                file = json.load(json_file) 
                                accepted = file.get('accepted')
                            y.append(accepted)
                        else:
                            y.append(True)
                assert len(x) == len(y), "data and label don\'t match."  
            return x, np.array(y)
        else:
            raise ValueError('mode doesn\'t exists')

    
    if os.path.exists(abs_path): 
        logging.info('Loading start...')
        path = {}
        path['acl'] = os.path.join(abs_path,'data/acl_2017')
        path['iclr'] = os.path.join(abs_path,'data/iclr_2017')
        path['coNLL'] = os.path.join(abs_path, 'data/conll_2016')
        path['ai'] = os.path.join(abs_path, 'data/arxiv.cs.ai_2007-2017')
        path['cl'] = os.path.join(abs_path, 'data/arxiv.cs.cl_2007-2017')
        path['lg'] = os.path.join(abs_path, 'data/arxiv.cs.lg_2007-2017')
        data_mode = {'train' : 'train/parsed_pdfs/',
                 'test' : 'test/parsed_pdfs/',
                 'dev' : 'dev/parsed_pdfs/'}
        label_mode = {'train' : 'train/reviews/',
                 'test' : 'test/reviews/',
                 'dev' : 'dev/reviews/'}

        meta, data, label = {}, {}, {}
        meta['acl'], data['acl'], label['acl'] = get_parsed(path['acl'], data_mode, label_mode)
        meta['iclr'], data['iclr'], label['iclr'] = get_parsed(path['iclr'], data_mode, label_mode)
        meta['coNLL'], data['coNLL'], label['coNLL'] = get_parsed(path['coNLL'], data_mode, label_mode)
        meta['ai'], data['ai'], label['ai'] = get_parsed(path['ai'], data_mode, label_mode)
        meta['cl'], data['cl'], label['cl'] = get_parsed(path['cl'], data_mode, label_mode)
        meta['lg'], data['lg'], label['lg'] = get_parsed(path['lg'], data_mode, label_mode)

        X_train_corpus, y_train = get_data(meta, data, label, data_mode, label_mode, 'train')
        logging.info('Training data loaded.')
        X_test_corpus, y_test = get_data(meta, data, label, data_mode, label_mode, 'test')
        logging.info('Testing data loaded.')

        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

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

        return X_train_corpus, X_test_corpus, y_train, y_test
    else:
        raise ValueError('Path doesn\'t exists...')

def load_arxiv(abs_path,
                   tokenize=True, 
                   lower=True, 
                   shuffle=True, 
                   random_state=42):
    """
    Load data for peerRead
    
    # Arguments
        tokenize: Boolean
        lower: Boolean
        shuffle: Boolean
        random_state: Integer
    # Returns
        X_train, X_test, y_train, y_test
    
    """
    
    import json
    
    def get_parsed(path, 
                   data_mode, 
                   label_mode):
        """
        get the file names under different conference folder
        
        # Arguments
            path: Path to the parent folder
            data_mode: dict.
                See below on data_mode (key,value) -> ('train', 'path to train')
            label_mode: dict.
                same as data. But on label to get reviews
        
        """
        parsed_data = {}
        parsed_label = {}

        for key,item in data_mode.items():
            parsed_data[key] = sorted(glob.glob1(os.path.join(path, item),
                                         '*'))
        
        for key,item in label_mode.items():
            parsed_label[key] = sorted(glob.glob1(os.path.join(path, item),
                                          '*'))
            
        meta = [(key, len(item)) for key,item in parsed_data.items()]

        return meta, parsed_data, parsed_label
    
    def get_data(meta, data, label, data_mode, label_mode, mode=None):
        """
        
        # Args
            mode: {'train', 'test', 'dev'}
        # returns
            x, y
        
        """
#         ignore = ['acl', 'iclr', 'coNLL']
        if mode is not None:
            x = []
            y = []
            for k in data.keys():
                for i in range(len(data[k][mode])):
                        # data
                        with open(os.path.join(path[k], data_mode[mode], data[k][mode][i])) as json_file:  
                            file = json.load(json_file) 
                            if file['metadata']['sections'] is not None:
                                temp = ' '
                                for key in file['metadata']['sections']:
                                    if key['heading'] is not None:
                                        temp = ' '.join([temp, '{} {}'.format(key['heading'], key['text'].strip())])
                        x.append(temp.replace('\n', ' '))

#                         if k not in ignore:
                        if k == 'ai':
                            y.append(1)
                        elif k == 'cl':
                            y.append(2)
                        elif k == 'lg':
                            y.append(3)
                assert len(x) == len(y), "data and label don\'t match."  
            return x, np.array(y)
        else:
            raise ValueError('mode doesn\'t exists')

    
    if os.path.exists(abs_path): 
        logging.info('Loading start...')
        path = {}
#         path['acl'] = os.path.join(abs_path,'data/acl_2017')
#         path['iclr'] = os.path.join(abs_path,'data/iclr_2017')
#         path['coNLL'] = os.path.join(abs_path, 'data/conll_2016')
        path['ai'] = os.path.join(abs_path, 'data/arxiv.cs.ai_2007-2017')
        path['cl'] = os.path.join(abs_path, 'data/arxiv.cs.cl_2007-2017')
        path['lg'] = os.path.join(abs_path, 'data/arxiv.cs.lg_2007-2017')
        data_mode = {'train' : 'train/parsed_pdfs/',
                 'test' : 'test/parsed_pdfs/',
                 'dev' : 'dev/parsed_pdfs/'}
        label_mode = {'train' : 'train/reviews/',
                 'test' : 'test/reviews/',
                 'dev' : 'dev/reviews/'}

        meta, data, label = {}, {}, {}
#         meta['acl'], data['acl'], label['acl'] = get_parsed(path['acl'], data_mode, label_mode)
#         meta['iclr'], data['iclr'], label['iclr'] = get_parsed(path['iclr'], data_mode, label_mode)
#         meta['coNLL'], data['coNLL'], label['coNLL'] = get_parsed(path['coNLL'], data_mode, label_mode)
        meta['ai'], data['ai'], label['ai'] = get_parsed(path['ai'], data_mode, label_mode)
        meta['cl'], data['cl'], label['cl'] = get_parsed(path['cl'], data_mode, label_mode)
        meta['lg'], data['lg'], label['lg'] = get_parsed(path['lg'], data_mode, label_mode)

        X_train_corpus, y_train = get_data(meta, data, label, data_mode, label_mode, 'train')
        logging.info('Training data loaded.')
        X_test_corpus, y_test = get_data(meta, data, label, data_mode, label_mode, 'test')
        logging.info('Testing data loaded.')

        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

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

        return X_train_corpus, X_test_corpus, y_train, y_test
    else:
        raise ValueError('Path doesn\'t exists...')

        
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
    neutral_indices = np.where(y_label == 3)[0]

    y[y < 3] = 0
    y[y > 3] = 1

    X_final = np.delete(X, neutral_indices)
    y_final = np.delete(y, neutral_indices)

    del X, y

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
        X_final = [text.lower() for text in X_final]
        logging.info('Lowered.')
        
    if tokenize:
        X_final = [word_tokenize(text) for text in X_final]
        logging.info('Tokenized.')

    
    if train_test_split:
        import sklearn

        if test_split is not None:
            test = test_split
        else:
            test = 0.33

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_final, 
                                                                                    y_final, 
                                                                                    test_size=test, 
                                                                                    random_state=random_state)

        return X_train, X_test, y_train, y_test
    else:
        return X_final, y_final
    

        
