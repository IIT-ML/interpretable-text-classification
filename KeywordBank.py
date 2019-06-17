"""
Implementation of KeywordBank. This is the parent class for any sub-class using keyword for next tasks.


Usage:
    >>> imdbKeyword = KeywordBank(keyword=keyword, 
    >>>                      xtrain=X_train_corpus, 
    >>>                      ytrain=y_train)
    >>> imdbKeyword.get_connotation()
    >>> imdbKeyword.connotation
    
Author: Anneke Hidayat, Mitchell Zhen, Mustafa Bilgic
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
            self.connotation = None
        
        if isinstance(xtrain, list):
            self.xtrain = xtrain
            self.ytrain = ytrain
        else:
            raise ValueError('Need reference data...')
        
    def get_connotation(self):
        """ Calculate sentiment score
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