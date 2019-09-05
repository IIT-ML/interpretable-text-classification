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
import tensorflow as tf
import keras
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

class InterpretableCautiousText():
    '''
    
    Assuming that we already have vectorized input of document,
    and vectorized input of keyword
    
    # Attributes
        doc_input_shape:
        keyword_input_shape: 

    # Methods
        building initial assessment
        building final classifier
    
    '''
    
    def __init__(self, doc_input_shape, keyword_input_shape):
        
        
        self.doc_input_shape = doc_input_shape
        self.keyword_input_shape = keyword_input_shape    
        
        self.initial_params = {'loss': 'binary_crossentropy',
            'metrics': ['acc'],
            'optimizer': 'adam'
        }
        
        self.final_params = {'loss': 'binary_crossentropy',
            'metrics': ['acc'],
            'optimizer': 'adam'
        }        
        
        self.initial_model = self.build_initial_assessment()
        self.initial_model.compile(loss=self.initial_params['loss'],
                                  metrics=self.initial_params['metrics'],
                                  optimizer=self.initial_params['optimizer'])
        

        self.final_model = self.build_final_classifier(self.initial_model)
        self.final_model.compile(loss=self.final_params['loss'],
                                  metrics=self.final_params['metrics'],
                                  optimizer=self.final_params['optimizer'])
        
        
    
    def build_initial_assessment(self):
        """
        Build initial assessment model that predicts the 
        document sentiment by it's whole input
        
        # Arguments
            input_shape: Shape tuple (integer). 
                Not including the batch size. 
                See: https://keras.io/layers/core/
                
        # Return
            model
        
        """
        initial_input = keras.layers.Input(shape=(self.doc_input_shape,),
                            name='initial_input')
        initial_output = keras.layers.Dense(1,
                                   activation='sigmoid',
                                   kernel_initializer=keras.initializers.glorot_uniform(seed=RAND_SEED))(initial_input)
        model = keras.models.Model(inputs=initial_input,
                     outputs=initial_output)
        return model
    
    def build_final_classifier(self, initial_model):
        """
        Build final classifier model that predicts the document
        sentiment by its justification
        
        # Arguments
        
        
        # Returns
        """
        # Initialization from the initial assessment layer
        final_input = keras.layers.Input(shape=(self.doc_input_shape,),
                                        name='final_input')
        initial_assessment_output = initial_model(final_input)
        hard_sigmoid_norm = keras.layers.Lambda(self._hard_sigmoid,
                                               name='initial_hard_sigmoid')(initial_assessment_output)
        
        # Keyword layer start here
        keyword_input = keras.layers.Input(shape=(self.keyword_input_shape,),
                                          name='keyword_input')
        keyword_split = keras.layers.Lambda(self._layer_split,
                                           name='keyword_split')(keyword_input)
        
        # Now the hard work is here
        # Stack the splitted layer
        initial_x_keyword = []
        for i in range(self.keyword_input_shape):
            initial_x_keyword.append(
                keras.layers.Lambda(self._relu)(keras.layers.Multiply()([keyword_split[i], hard_sigmoid_norm])))
        # concatenate layers
        final_concat = keras.layers.Lambda(self._layer_concat, name='final_concatenate')(initial_x_keyword)
        
        # final prediction
        final_output = keras.layers.Dense(1,
                                         activation='sigmoid',
                                         name='final_output')(final_concat)
        
        model = keras.models.Model(inputs=[final_input, keyword_input],
                     outputs=final_output)
        return model
        
    
    # Helper lambda layer start here
    def _hard_sigmoid(self, x):
        return (x*2)-1
        
    def _relu(self, x):
        return tf.nn.relu(x)
    
    def _layer_split(self,x):
        """
        split each unit in one layer into single layer with one unit each
        """
        return tf.split(x,num_or_size_splits=self.keyword_input_shape,axis=1)

    def _layer_concat(self,x):
        return tf.concat(x, axis=1)
    
def train(model, 
          data, 
          label, 
          epochs,
          batch_size,
          lr,
          lr_decay,
          weights_dir,
          train_mode=3, 
          save_weights=True,
         return_callbacks=True):
    """ Train Interpretable and Cautious Model
    
    Helper to train it in a different fashion
    
    # Arguments
        model:
        data:
        label:
        train_mode: {'default-joint', 'pre-trained-frozen', 'pre-trained-joint'}
            Mode on how we train the model. Refer to the paper regarding the training methods

    """
    
    # callbacks
    if return_callbacks:
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
        #log = keras.callbacks.CSVLogger(os.path.join(log_dir, 'log-{}.csv'.format(train_mode)))
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(weights_dir, '{epoch:03d}-{val_loss:.3f}-{val_acc:.3f}.h5'),
                                                    monitor='val_loss',
                                                    mode='min',
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    verbose=1)
        #lr_decay = keras.callbacks.LearningRateScheduler(schedule=lambda epoch: lr * (lr_decay ** float(epoch)))
        callbacks = [early_stop, checkpoint]
    else:
        callbacks = None
    
    if train_mode == 1:
        # Train the model altogether {default-joint}
        model.final_model.fit([data['docs'], data['keys']],
                             label,
                             validation_split = (1./3),
                             shuffle=True,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=callbacks)
        
    elif train_mode == 2:
        # pre-trained frozen
        model.initial_model.fit(data['docs'],
                               label,
                               validation_split = (1./3),
                               shuffle=True,
                               batch_size=batch_size,
                               epochs=1)
        
        # Froze the initial model's weights
        model.initial_model.trainable=False
        
        model.final_model.fit([data['docs'], data['keys']],
                             label,
                             validation_split = (1./3),
                             shuffle=True,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=callbacks)
        
    elif train_mode == 3:
        # ore-trained joint
        model.initial_model.fit(data['docs'],
                               label,
                               validation_split = (1./3),
                               shuffle=True,
                               batch_size=batch_size,
                               epochs=1)
        
        # Make it trainable when it trained the final model
        model.initial_model.trainable=True
        
        model.final_model.fit([data['docs'], data['keys']],
                             label,
                             validation_split = (1./3),
                             shuffle=True,
                             batch_size=batch_size,
                             epochs=epochs,
                             callbacks=callbacks)
    else:
        raise ValueError('train_mode is not recognized..')
    
    #if save_weights:
    #    model.final_model.save_weights(os.path.join(weights_dir, '{}-trained-model.h5'.format(train_mode)))
        
    #if epochs > 1:
    #    from utils import plot_log
    #    plot_log(os.path.join(LOG_DIR, 'log-{}.csv'.format(train_mode)),
    #             show=True)
    
    return model

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
    parser.add_argument('--train_mode', 
                        default=1, 
                        type=int,
                        help="1:default-joint, 2:pre-trained-frozen, 3:pre-trained-joint")
    
    #### Let's make the directory placement automatic
    parser.add_argument('--log_dir', 
                        default='./log', 
                        type=str,
                        help="dir to save log and summary")   
    parser.add_argument('-w', '--weights', 
                        default=None,
                        help="The path of the saved weights. Specified when testing")
    parser.add_argument('-t', '--testing', 
                        action='store_true',
                        help="test data with the given model path")
    parser.add_argument('--epochs', 
                        default=1, 
                        type=int)
    parser.add_argument('--batch_size', 
                        default=1, 
                        type=int)
    parser.add_argument('--lr', 
                        default=0.001, 
                        type=float)
    parser.add_argument('--lr_decay', 
                        default=0.9, 
                        type=float)
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
        KEYWORD_PATH = '/home/anneke/Documents/ann-mitchell-text-classification/data/agnews-sci_sport-keywords/agnews_scitechworld_keywords.json'
        
        print('{}: Load dataset'.format(config['start_time']))
        
        if os.path.exists(DATA_PATH) and os.path.exists(KEYWORD_PATH):
            keyword = json.load(open(KEYWORD_PATH, 
                                     'r'))
            
            print('Loading...')
            
            X_train, X_test, y_train, y_test = load_ag_news(DATA_PATH,
                                                            shuffle = True,
                                                            lower = True,
                                                            tokenize = True)
            
            keywordObj = KeywordBank(keyword=keyword, 
                                     xtrain=X_train, 
                                     ytrain=y_train)
            
            keywordObj.assign_connotation(words_len=args.word_len, 
                                          class_label=['world', 'scitech'])
            
            print('vectorize...')
            X_train, X_test = utils.vectorize_keywords_docs(X_train, 
                                                            X_test, 
                                                            keywordObj)
            
            def get_sci_sports(X, y):
                ind = np.array(list(np.where(y==1)[0]) + list(np.where(y==4)[0]))
                
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
        print('Training....')
        
        directory = 'int-{}-{}-{}-{}-{}-{}'.format(args.dataset, 
                                             args.train_mode,
                                             args.word_len,
                                             args.epochs,
                                             args.batch_size,
                                             config['start_time'])
    
        w_dir = directory
        #l_dir = 'log_dir/{}'.format(directory)
        
        if not os.path.exists(os.path.join(args.parent_dir, w_dir)):
            os.mkdir(os.path.join(args.parent_dir, w_dir))
#         if not os.path.exists(os.path.join(args.parent_dir, l_dir)):
#             os.mkdir(os.path.join(args.parent_dir, l_dir))

        
        model = InterpretableCautiousText(X_train['docs'].shape[1], len(keywordObj.connotation))
        
        m = train(model, X_train,
                  y_train,
                  args.epochs,
                  args.batch_size,
                  args.lr,
                  args.lr_decay, 
                  os.path.join(args.parent_dir, w_dir),
                  args.train_mode)
        
        config['end_time'] = datetime.datetime.now(timezone('US/Central')).strftime("%y-%m-%d_%H:%M:%S")
        
        best_model = sorted(glob.glob(os.path.join(args.parent_dir, w_dir)+"/*.h5"))[-1]
        
        print(best_model)
        m.final_model.load_weights(best_model)
        
        report, preds, exp_vec = test(m,
                                      X_test,
                                      label=y_test)
        
        config['report'] = report
        config['keyword_list'] = sorted(list(keywordObj.connotation.keys()))
        with open('{}/CONFIG'.format(os.path.join(args.parent_dir, w_dir)), 'w') as outfile:
            json.dump(config, outfile, indent=4)
           
        l = [v for v in exp_vec]
        df = pd.DataFrame({'label': y_test,
                           'preds': preds,
                           'explanation':l})
        
        df.to_parquet('{}/test.parquet'.format(os.path.join(args.parent_dir, w_dir)))
        print('Finish.....')
    else:
        # load weights
        pre_trained_weight = args.weights

        report, preds, exp_vec = test(m,
                              X_test, 
                              label=y_test)
        
        #TODO: Create file. Let's create on df. since we have exp_vec. 
        #label, preds, exp_vec
        print(report)

        
