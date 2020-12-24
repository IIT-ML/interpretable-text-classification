"""
Implementation of KeywordBank. This is the parent class for any sub-class using keyword for next tasks.


Usage:
    >>> imdbKeyword = KeywordBank(keyword=keyword, 
    >>>                      xtrain=X_train_corpus, 
    >>>                      ytrain=y_train)
    >>> imdbKeyword.get_connotation()
    >>> imdbKeyword.connotation
    
"""

import os

class KeywordBank():
    """ Keyword object. Use as parent class
    
    # Attribute
        keyword: list
        reference_data: list (already tokenized)
        
    # Methods
        get_connotation
    """
    
    def __init__(self, 
                 keyword=None, 
                 xtrain=None, 
                 ytrain=None,
                 connotation=None):
        
        self.keyword = keyword
        
        if connotation is not None:
            self.connotation = connotation
        else:
            self.connotation = {}
        
        if isinstance(keyword, list):
            if isinstance(xtrain, list):
                self.xtrain = xtrain
                self.ytrain = ytrain
            else:
                raise ValueError('Need reference data...')
        else:
            self.xtrain = xtrain
            self.ytrain = ytrain
    
    def assign_connotation(self, words_len='100', class_label=['neg', 'pos']):
        """
            Use this function only to generate connotation dictionary
            from the given dictionary
            
            # Arguments
            words_length    str
                            {'100', '200', '300'}
            class_label     class index for each dataset. 
                            It has to map between classes in the keyword file
        """
        # make sure the keywords are dictionary
        # all the words are belongs to binary classes
            
        for c in class_label:
            for w in self.keyword[words_len][c]:
                self.connotation[w] = 1 if class_label.index(c) == 1 else -1
    
    def generate_connotation(self):
        """ We may use this function only if we don't have any connotation given
            as an input. 
            Calculate sentiment score
            Use training data only. Calculate as follow :
            if f_{+}(keyword) > f_{-}(keyword):
                +1
            else:
                -1
                
        # Arguments
            
        """
        if self.xtrain is None:
            raise ValueError('Need training data '
                            'for inspect connotation')
                
        if self.keyword is not None:
            self.connotation = {}
            for key in self.keyword:
                pos_count = 0
                neg_count = 0
                for i, doc in enumerate(self.xtrain):
                    if key in doc:
                        if (self.ytrain[i] == 1):
                            pos_count += 1
                        else:
                            neg_count += 1
                    
                if pos_count > neg_count:
                    self.connotation[key] = 1
                else:
                    self.connotation[key] = -1
        else:
            raise ValueError('Keyword does not exist')       