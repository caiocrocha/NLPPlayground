class CustomPortugueseLemmatizer():
    """Canonicalize a dataset by applying lemmatization to texts
    in the portuguese language with Spacy's "pt_core_news_sm" module
    """
    def __init__(self):
        import spacy
        self.spacy_nlp = spacy.load("pt_core_news_sm")
    
    # Returns True if the word is in CV (consonant-vowel) format or False if it is not
    def is_canonical(self, word):
        canonical = True
        seq = iter(word)

        for c in seq:
            if c not in 'bcdfghjklmnpqrstvwxyzç':
                canonical = False
                break
            else:
                try:
                    n = next(seq)
                except StopIteration as e:
                    canonical = False
                    break
                if n not in 'aeiouáàãâéêíóõôúü':
                    canonical = False
                    break
        return canonical
    
    def remove_case(self, token):
        return (
            token.is_stop 
            or any(c.isdigit() for c in token.text) 
            or token.pos_ == 'PUNCT' 
            or token.pos_ == 'NUM' 
            or token.pos_ == 'SPACE' 
            or token.pos_ == 'SYM' 
            or token.pos_ == 'X'
               )
    
    def fit(self, raw_documents, y=None):
        return self
    
    def transform(self, raw_documents):
        X = []
        for text in raw_documents:
            word_list = []
            cv_list = []
            for token in self.spacy_nlp(text):
                # only append useful words, excluding stop words, numbers, 
                # spaces, punctuations, symbols and unknown characters
                if not self.remove_case(token): 
                    word = token.lemma_.lower()
                    word_list.append(word)
                    # assign if word is in CV (consonant-vowel) format or if it is not defined
                    # word = word + '_cv' if is_canonical(word) else word + '_nd'
                    cv_list.append('is_cv' if self.is_canonical(word) else 'not_cv')

            sentence = ' '.join(word_list)
            sentence += ' '
            sentence += ' '.join(cv_list)
            X.append(sentence)
        return X
    
    def fit_transform(self, raw_documents, y=None):
        return self.fit(raw_documents, y).transform(raw_documents)