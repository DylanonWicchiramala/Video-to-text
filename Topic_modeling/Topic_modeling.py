import spacy

class Topic_modeling:
    
    def __init__(self, documents):
        self.preprocessor = spacy.load('en_core_web_sm')
    
    def __text_preprocess(self, text, lemmatization=True, return_list=True):

        #creating doc object containing our token features
        doc = self.preprocessor(text)

        # tokenization, lemmatization and removing stop words.
        if lemmatization:
            tokens = [token.lemma_ for token in doc if not(token.is_stop) and not(token.is_punct)]
        else:
            tokens = [token.text for token in doc if not(token.is_stop) and not(token.is_punct)]

        if return_list:
            return tokens
        else:
            return " ".join(tokens)
        
    
    def get_document(self):
        pass
    
    def extract_topic(self):
        pass
    
    def get_topic(self):
        pass
    
    