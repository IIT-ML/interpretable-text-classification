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
import glob
import datetime
import pandas as pd
from pytz import timezone

from utils import utils, dataset_helper
from KeywordBank import KeywordBank
from nltk.tokenize import word_tokenize
from utils.dataset_helper import load_ag_news
RAND_SEED = 42

def test(model, 
         data, 
         label=None, 
         threshold=.5):
    """ Test it with rejection rate
    
    # Arguments
        model: InterpretableCautiousText object
        data:
        label:
        threshold:
        
    # Returns
        For each document,
            ['label', 'explanation']
            label: class
            explanation: keyword vector (zero and non-zero)
    """
    md = model.final_model
    
    # Workflow:
    # 1. Get the explanation vector model 
    #    after multiplied with initial prediction
    # 2. Get the explanation vector
    # 3. Calculate rejection rate
    # 4. Final prediction on document with explanation
    # 5. Return report on the result along with the explanation vector
    report = {}
        
    explanation_model = keras.models.Model(inputs=md.input,
                                   outputs=md.get_layer('final_concatenate').output)
    explanation_vec = explanation_model.predict([data['docs'], data['keys']])
        
    doc_YES_exp_indices = np.squeeze(np.where(np.sum(explanation_vec, axis=1)!=0))
    doc_NO_exp_indices = np.squeeze(np.where(np.sum(explanation_vec, axis=1)==0))
    
    preds = md.predict([data['docs'], data['keys']])
    preds = np.array([1. if d>threshold else 0. for d in preds])
    preds[doc_NO_exp_indices] = -1
        
    rejection_rate = len(doc_NO_exp_indices) / data['docs'].shape[0]
        
    report['rejection_rate'] = np.around(rejection_rate, 3)
    report['total_reject'] = len(doc_NO_exp_indices)
    report['total_accept'] = len(doc_YES_exp_indices)
        
    if label is None:
        return report, preds, explanation_vec
    else:
        if isinstance(label, pd.DataFrame):
            eval_result = md.evaluate([data['docs'][doc_YES_exp_indices], data['keys'][doc_YES_exp_indices]],
                                       label.iloc[doc_YES_exp_indices].get_values())
        else:
            if isinstance(label, list):
                lable = np.array(label)
            
            eval_result = md.evaluate([data['docs'][doc_YES_exp_indices], data['keys'][doc_YES_exp_indices]],
                                       label[doc_YES_exp_indices])
        
        report['loss'] = np.around(eval_result[0], 3)
        report['acc'] = np.around(eval_result[1], 3)
            
        return report, preds, explanation_vec



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
    parser.add_argument('-t', '--testing', 
                        action='store_true',
                        help="test data with the given model path")
    parser.add_argument('--parent_dir', 
                        default='/home/anneke/Documents/models/', 
                        type=str,
                        help="Path to save model")
    parser.add_argument('--word_len',
                        default='100',
                        type=str)
    
    
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
            
            config['data_summary']['keyword'] = keyword[args.word_len]['summary']
            config['data_summary']['data'] = {'train':len(y_train), 
                                              'test':len(y_test)}
        else:
            raise ValueError('Path doesn\'t exist. Please check the availability of your data')
    elif args.dataset.lower() == 'arxiv':
        DATA_PATH = '/home/anneke/Documents/ann-mitchell-text-classification/dataset/arxiv_ai_crypto_data.parquet'
        KEYWORD_PATH = '/home/anneke/Documents/ann-mitchell-text-classification/data/arxiv-aicrypto-keywords/arxiv_keywords.json'
        
        print('{}: Load dataset'.format(config['start_time']))
        
        if os.path.exists(DATA_PATH) and os.path.exists(KEYWORD_PATH):
            keyword = json.load(open(KEYWORD_PATH, 
                                     'r'))
            
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
            
            keywordObj = KeywordBank(keyword=keyword, 
                                     xtrain=X_train, 
                                     ytrain=y_train)
            
            keywordObj.assign_connotation(words_len=args.word_len, 
                                          class_label=['crypto', 'ai'])
            
            print('vectorize...')
            X_train, X_test = utils.vectorize_keywords_docs(X_train, 
                                                            X_test, 
                                                            keywordObj)
            config['data_summary']['keyword'] = keyword[args.word_len]['summary']
            config['data_summary']['data'] = {'train':len(y_train), 
                                              'test':len(y_test)}
        else:
            raise ValueError('Path doesn\'t exist. Please check the availability of your data')
    elif args.dataset.lower() == 'agnews':
        DATA_PATH = '/home/anneke/Documents/ann-mitchell-text-classification/dataset/ag_news_csv/'
        KEYWORD_PATH = '/home/anneke/Documents/ann-mitchell-text-classification/data/agnews-sci_sport-keywords/agnews_keywords.json'
        
        print('{}: Load dataset'.format(config['start_time']))
        
        if os.path.exists(DATA_PATH) and os.path.exists(KEYWORD_PATH):
            keyword = json.load(open(KEYWORD_PATH, 
                                     'r'))
            
            print('{}-{}-Loading....'.format(args.dataset, args.word_len))
            
            X_train, X_test, y_train, y_test = load_ag_news(DATA_PATH,
                                                            shuffle = True,
                                                            lower = True,
                                                            tokenize = True)
            
            keywordObj = KeywordBank(keyword=keyword, 
                                     xtrain=X_train, 
                                     ytrain=y_train)
            
            keywordObj.assign_connotation(words_len=args.word_len, 
                                          class_label=['sports', 'scitech'])
            
            print('{}-{}-Vectorize....'.format(args.dataset, args.word_len))
            X_train, X_test = utils.vectorize_keywords_docs(X_train, 
                                                            X_test, 
                                                            keywordObj)
            
            def get_sci_sports(X, y):
                ind = np.array(list(np.where(y==2)[0]) + list(np.where(y==4)[0]))
                
                X['docs'] = X['docs'][ind]
                X['keys'] = X['keys'][ind]
                y = y[ind]

                y = [1 if y_==4 else 0 for y_ in y]
                
                return X, np.array(y)
            
            X_train, y_train = get_sci_sports(X_train, y_train)
            X_test, y_test = get_sci_sports(X_test, y_test)
            
            config['data_summary']['keyword'] = keyword[args.word_len]['summary']
            config['data_summary']['data'] = {'train':len(y_train), 
                                              'test':len(y_test)}
    else:
        
        # TODO: add if there is any directory to new dataset.
        pass
    
    # Train / test
    
    if not args.testing:
        
        directory = 'base-{}-{}-{}'.format(args.dataset, 
                                             args.word_len,
                                             config['start_time'])
        
        w_dir = 'weights/{}'.format(directory)
        
        if not os.path.exists(os.path.join(args.parent_dir, w_dir)):
            os.mkdir(os.path.join(args.parent_dir, w_dir))
            
        print('{}-{}-LRs....'.format(args.dataset, args.word_len))
        from sklearn.linear_model import LogisticRegression
        
        
        clf = LogisticRegression(penalty='l1', random_state=RAND_SEED)
        clf.fit(X_train['docs'], y_train)
        config['results'] = {}
        
        config['results']['LR'] = {}
        config['results']['LR']['train_acc'] = clf.score(X_train['docs'], y_train)
        config['results']['LR']['test_acc'] = clf.score(X_test['docs'], y_test)
        
        del clf

        # rejection rate count
        config['results']['LR_keys'] = {}
        config['results']['LR_keys']['train']= {}
        config['results']['LR_keys']['test']= {}
        
        # change all connotation to 1
        tr_row_indices, tr_feat_indices = X_train['keys'].nonzero()
        te_row_indices, te_feat_indices = X_test['keys'].nonzero()
        
        for row, feat in zip(tr_row_indices, tr_feat_indices):
            X_train['keys'][row, feat] = 1
        for row, feat in zip(te_row_indices, te_feat_indices):
            X_test['keys'][row, feat] = 1
            
        print('{}-{}-LR Keys....'.format(args.dataset, args.word_len))
        
        clf = LogisticRegression(penalty='l1', random_state=RAND_SEED)
        clf.fit(X_train['keys'], y_train)
        
        w_sum = np.sum(X_train['keys'].todense(), axis=1)
        accept_indices = np.squeeze(np.where(w_sum != 0))[0]
        reject_indices = np.squeeze(np.where(w_sum == 0))[0]
        
        config['results']['LR_keys']['train']['total_accept'] = len(accept_indices)
        config['results']['LR_keys']['train']['total_reject'] = len(reject_indices)
        config['results']['LR_keys']['train']['rejection_rate'] = len(reject_indices) / len(y_train)
       
        try:
            config['results']['LR_keys']['train']['accuracy_with_reject'] = clf.score(X_train['keys'][accept_indices],
                                                                                     y_train[accept_indices])
        except:
            y_train = np.array(y_train.tolist())
            config['results']['LR_keys']['train']['accuracy_with_reject'] = clf.score(X_train['keys'][accept_indices],
                                                                                     y_train[accept_indices])
            
        config['results']['LR_keys']['train']['accuracy'] = clf.score(X_train['keys'], y_train)
        
        w_sum = np.sum(X_test['keys'].todense(), axis=1)
        accept_indices = np.squeeze(np.where(w_sum != 0))[0]
        reject_indices = np.squeeze(np.where(w_sum == 0))[0]
        
        config['results']['LR_keys']['test']['total_accept'] = len(accept_indices)
        config['results']['LR_keys']['test']['total_reject'] = len(reject_indices)
        config['results']['LR_keys']['test']['rejection_rate'] = len(reject_indices) / len(y_train)
        
        try:
            config['results']['LR_keys']['test']['accuracy_with_reject'] = clf.score(X_test['keys'][accept_indices],
                                                                                     y_test[accept_indices])
        except:
            y_test = np.array(y_test.tolist())
            config['results']['LR_keys']['test']['accuracy_with_reject'] = clf.score(X_test['keys'][accept_indices],
                                                                                     y_test[accept_indices])
            
        config['results']['LR_keys']['test']['accuracy'] = clf.score(X_test['keys'], y_test)
        

        config['keyword_list'] = sorted(list(keywordObj.connotation.keys()))
        
        config['end_time'] = datetime.datetime.now(timezone('US/Central')).strftime("%y-%m-%d_%H:%M:%S")
        
        with open('{}/CONFIG'.format(os.path.join(args.parent_dir, w_dir)), 'w') as outfile:
            json.dump(config, outfile, indent=4)
           
        
        print('{}-{}-Finish....'.format(args.dataset, args.word_len))
    else:
        # load weights
        pre_trained_weight = args.weights

        report, preds, exp_vec = test(m,
                              X_test, 
                              label=y_test)
        
        #TODO: Create file. Let's create on df. since we have exp_vec. 
        #label, preds, exp_vec
        print(report)

        
