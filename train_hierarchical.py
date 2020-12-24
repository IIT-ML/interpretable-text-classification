"""
Script to the model. 

Class:
    InterpretableCautiousText

Method:
    train
    test

Usage:
    >>> model = InterpretableCautiousText()
  

"""

import os
import tensorflow as tf
import keras
from keras import backend as K


import numpy as np
import json
import glob
import datetime
import pandas as pd
from textblob import TextBlob
from pytz import timezone
from keras.preprocessing.text import Tokenizer, text_to_word_sequence



from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers import CuDNNLSTM, CuDNNGRU
# Merge
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
# from keras import initializations
from keras import initializers
from keras import regularizers
from keras import optimizers
from keras import constraints

from utils import utils, dataset_helper
from KeywordBank import KeywordBank
from nltk.tokenize import word_tokenize
from utils.dataset_helper import load_ag_news
RAND_SEED = 42

MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

np.random.seed(RAND_SEED)
tf.set_random_seed(RAND_SEED)


def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    
class AttentionWithContext(keras.layers.Layer):
    def __init__(self,
        W_regularizer=None, u_regularizer=None, b_regularizer=None,
        W_constraint=None, u_constraint=None, b_constraint=None,
        bias=True, **kwargs):
            
        self.supports_masking = False
        self.init = keras.initializers.get('normal')
            
        self.W_regularizer = keras.regularizers.get(W_regularizer)
        self.u_regularizer = keras.regularizers.get(u_regularizer)
        self.b_regularizer = keras.regularizers.get(b_regularizer)
            
        self.W_constraint = keras.constraints.get(W_constraint)
        self.u_constraint = keras.constraints.get(u_constraint)
        self.b_constraint = keras.constraints.get(b_constraint)
    
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                initializer=self.init,
                                name='{}_W'.format(self.name),
                                regularizer=self.W_regularizer,
                                constraint=self.W_constraint)
            
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                    initializer='zero',
                                    name='{}_b'.format(self.name),
                                    regularizer=self.b_regularizer,
                                    constraint=self.b_constraint)
            
        self.u = self.add_weight((input_shape[-1],),
                                initializer=self.init,
                                name='{}_u'.format(self.name),
                                regularizer=self.u_regularizer,
                                constraint=self.u_constraint)
    
        super(AttentionWithContext, self).build(input_shape)
    
    def compute_mask(self, input, input_mask=None):
        return None
        
    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
            
        if self.bias:
            uit += self.b
                
        uit = K.tanh(uit)
#         ait = K.dot(uit, self.u) # only works on  
        
        ait = dot_product(uit, self.u)
        a = K.exp(ait)
        
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
            
        a = K.expand_dims(a)
        weighted_input = x * a
            
        return K.sum(weighted_input,axis=1)
        
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

if __name__ == "__main__":
    import argparse

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Interpretable Cautious Text Classifier args")
    parser.add_argument('--gpu', 
                        action='store_true',
                        help="If given, the operation will be operated in GPU")
    
    #### For temporary, the amazon_video and e_commerce won't be available
    parser.add_argument('--dataset', 
                        default='imdb',
                        help="dataset. {'imdb', 'arxiv', 'agnews'}. If path given, use the path")
    parser.add_argument('--model', 
                        default='HAN',
                        help="dataset. {'imdb', 'arxiv', 'agnews'}. If path given, use the path")
    parser.add_argument('-t', '--testing', 
                        action='store_true',
                        help="test data with the given model path")
    parser.add_argument('--parent_dir', 
                        default='./models/', 
                        type=str,
                        help="Path to save model")
    
    
    args = parser.parse_args()
    # print(args)
    
    config = {}
    config['args'] = vars(args)
    config['start_time'] = datetime.datetime.now(timezone('US/Central')).strftime("%y-%m-%d_%H:%M:%S")
    config['data_summary'] = {}
    
    #format(datetime.datetime.now(timezone('US/Central')).strftime("%y%m%d_%H%M%S"))
    
    # GPU placement
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES']='0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        
   
    # Load dataset
    if args.dataset.lower() == 'imdb':
        # 1. Load keyword from txt file
        # 2. Load dataset
        DATA_PATH = './dataset/aclImdb'
        
        print('{}: Load dataset'.format(config['start_time']))
        
        if os.path.exists(DATA_PATH):
            
            print('Loading...')
            X_train, y_train, X_test, y_test = dataset_helper.load_imdb(DATA_PATH, 
                                                                                      lower=True, 
                                                                                      tokenize=True)
            
            config['data_summary']['data'] = {'train':len(y_train), 
                                              'test':len(y_test)}
        else:
            raise ValueError('Path doesn\'t exist. Please check the availability of your data')
    elif args.dataset.lower() == 'arxiv':
        DATA_PATH = './dataset/arxiv_ai_crypto_data.parquet'
        
        print('{}: Load dataset'.format(config['start_time']))
        
        if os.path.exists(DATA_PATH):
            
            print('Loading...')
            
            X_ = pd.read_parquet(DATA_PATH)
            
            def apply_categories(data, 
                                 labels=['cs.ai', 'cs.cr']):
                '''
                    Need to make sure that there is no overlap between these categories first!
                '''

                for l in labels:
                    if l in data.split(' '):
                        return l

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
            
            y_train = list(y_train)
            y_test = list(y_test)

            config['data_summary']['data'] = {'train':len(y_train), 
                                              'test':len(y_test)}
            
            print(config['data_summary']['data'])
        else:
            raise ValueError('Path doesn\'t exist. Please check the availability of your data')
            
    elif args.dataset.lower() == 'agnews':
        DATA_PATH = './dataset/ag_news_csv/'
        
        print('{}: Load dataset'.format(config['start_time']))
        
        if os.path.exists(DATA_PATH):
            
            X_train, X_test, y_train, y_test = load_ag_news(DATA_PATH,
                                                            shuffle = True,
                                                            lower = True,
                                                            tokenize = False)

            
            def get_sci_sports(X, y):
                ind = np.array(list(np.where(y==2)[0]) + list(np.where(y==4)[0]))
                
                X = [X[i] for i in ind]
                y = y[ind]

                y = [1 if y_==4 else 0 for y_ in y]
                
                return X, np.array(y)
            
            X_train, y_train = get_sci_sports(X_train, y_train)
            X_test, y_test = get_sci_sports(X_test, y_test)
            
            X_train = [word_tokenize(x) for x in X_train]
            X_test = [word_tokenize(x) for x in X_test]
            
            config['data_summary']['data'] = {'train':len(y_train), 
                                              'test':len(y_test)}
    else:
        print('Failed')
        # TODO: add if there is any directory to new dataset.
        pass
    
    # Train / test
    
    if not args.testing:
        
        # Tokenize
        X_tr_sentences = [TextBlob(' '.join(r)).raw_sentences for r in X_train]
        X_te_sentences = [TextBlob(' '.join(r)).raw_sentences for r in X_test]
        
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.fit_on_texts(X_train)

        data = np.zeros((len(X_train), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
        test_data = np.zeros((len(X_test), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
        
        # numerize token in Train data
        print('start tokenize train...')
        for i, sentences in enumerate(X_tr_sentences):
            for j, sent in enumerate(sentences):
                if j<MAX_SENTS:
                    wordTokens = keras.preprocessing.text.text_to_word_sequence(sent)
                    k=0
                    for _, word in enumerate(wordTokens):
                        try:
                            if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                                data[i,j,k] = tokenizer.word_index[word]
                                k += 1
                        except KeyError:
                            continue

        # numerize token in Test data
        print('start tokenize test...')
        for i, sentences in enumerate(X_te_sentences):
            for j, sent in enumerate(sentences):
                if j<MAX_SENTS:
                    wordTokens = keras.preprocessing.text.text_to_word_sequence(sent)
                    k = 0
                    for _,word in enumerate(wordTokens):
                        try:
                            if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                                test_data[i,j,k] = tokenizer.word_index[word]
                                k += 1
                        except KeyError:
                            continue
        
        word_index = tokenizer.word_index
        labels = y_train
        labels_test = y_test
        
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = [labels[i] for i in indices]
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
        
        x_train = data[:-nb_validation_samples]
        y_train = np.array(labels[:-nb_validation_samples])
        x_val = data[-nb_validation_samples:]
        y_val = np.array(labels[-nb_validation_samples:])

        x_test = test_data
        y_test = np.array(labels_test)
        
        ###### Embedding
        GLOVE_DIR = "/dataset/"
        embeddings_index = {}
        f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), 'rb')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Total %s word vectors.' % len(embeddings_index))
        
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        embedding_layer = keras.layers.Embedding(len(word_index)+1,
                                       EMBEDDING_DIM,
                                       weights=[embedding_matrix],
                                       input_length=MAX_SENT_LENGTH,
                                       trainable=True)        
        
        
        if args.model == 'HN':
            sentence_input = keras.layers.Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
            embedded_sequences = embedding_layer(sentence_input)
            l_gru = keras.layers.Bidirectional(keras.layers.GRU(100))(embedded_sequences)
            sentEncoder = keras.models.Model(sentence_input, l_gru)

            # Sentence
            review_input = keras.layers.Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
            review_encoder = keras.layers.TimeDistributed(sentEncoder)(review_input)
            l_gru_sent = keras.layers.Bidirectional(keras.layers.GRU(100))(review_encoder)

            preds = keras.layers.Dense(1, activation='sigmoid')(l_gru_sent)
            model = keras.models.Model(review_input, preds)

        else:
            sentence_input = keras.layers.Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
            embedded_sequences = embedding_layer(sentence_input)

            l_gru = keras.layers.Bidirectional(keras.layers.GRU(50, return_sequences=True))(embedded_sequences)
            l_dense = keras.layers.TimeDistributed(keras.layers.Dense(10))(l_gru)
            l_att = AttentionWithContext()(l_dense)

            sentEncoder = keras.models.Model(sentence_input, l_att)

            review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
            review_encoder = keras.layers.TimeDistributed(sentEncoder)(review_input)

            l_gru_sent = keras.layers.Bidirectional(keras.layers.GRU(50, return_sequences=True))(review_encoder)
            l_dense_sent = keras.layers.TimeDistributed(keras.layers.Dense(10))(l_gru_sent)
            l_att_sent = AttentionWithContext()(l_dense_sent)

            preds = keras.layers.Dense(1, activation='sigmoid')(l_att_sent)
            model = keras.models.Model(review_input, preds)
        
        
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
        
        model.fit(x_train, 
                  y_train, 
                  validation_data=(x_val, y_val),
                  epochs=1, 
                  batch_size=1,
                  verbose=1)    
        
        
        
        PREDS_PATH = './dataset/vectors'
        
        directory = '{}-{}-{}'.format(args.model,
                                      args.dataset, 
                                      config['start_time'])
        
        w_dir = 'weights/{}'.format(directory)
        
        if not os.path.exists(os.path.join(args.parent_dir, w_dir)):
            os.mkdir(os.path.join(args.parent_dir, w_dir))

        config['results'] = {}
        
        config['results']['train'] = model.evaluate(x_train, y_train)
        config['results']['test'] = model.evaluate(x_test, y_test)
        
        model.save_weights('{}/model.h5'.format(os.path.join(args.parent_dir, w_dir)))
        
        p = model.predict(x_test)
        p = [1 if p_ >= 0.5 else 0 for p_ in p]
        pd.DataFrame({'y_test':p}).to_parquet(os.path.join(PREDS_PATH, 
                                                           '{}-{}-LR-keys.parquet'.format(args.dataset,
                                                                                          args.model)))
        
        
        config['end_time'] = datetime.datetime.now(timezone('US/Central')).strftime("%y-%m-%d_%H:%M:%S")
        
        with open('{}/CONFIG'.format(os.path.join(args.parent_dir, w_dir)), 'w') as outfile:
            json.dump(config, outfile, indent=4)
           
        
        print('Finish....')