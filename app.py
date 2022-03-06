#!/usr/bin/env python3
# coding: utf-8
# Config

import streamlit as st
import random
import numpy as np
import spacy

import pickle
import os

# Program
class App():
    def __init__(self):
        # Show logo, title and description
        self.show_description()
        self.get_text()
        self.load_spacy()
        self.get_classifier()
        self.get_operation()
        self.apply_operation()
        self.load_pipeline()

    class CustomPortugueseLemmatizer():
        """Canonicalize a dataset by applying lemmatization to texts
        in the portuguese language with Spacy's "pt_core_news_sm" module
        """
        def __init__(self, spacy_nlp):
            self.spacy_nlp = spacy_nlp
        
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
                for token in self.nlp(text):
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

    @staticmethod
    def show_description():
        st.markdown('''## **PlayGround de NLP**
# Como está a minha escrita?
## Um classificador automático do nível de escrita de um texto
Escreva um pequeno texto (0 a 500 palavras) sobre um assunto específico e descubra se seu perfil de escrita se encaixa nos níveis do Ensino Fundamental I, Ensino Fundamental II, Ensino Médio ou Ensino Superior.
Você também pode descobrir como operações de NLP (Natural Language Processing) podem influenciar na classificação, por meio dos controles na barra lateral!
''')

    def get_text(self):
        self.text = st.text_area('Escreva um texto e aperte Ctrl+Enter para enviar.')
        self.original_text = self.text

    def load_spacy(self):
        self.spacy_nlp = spacy.load("pt_core_news_sm")

    def get_classifier(self):
        models = ['Naive Bayes (NB)', 'Support Vector Classifier (SVC)']
        name = st.sidebar.selectbox(label="Classificador", options=models)

        if name == 'Naive Bayes (NB)':
            from sklearn.naive_bayes import MultinomialNB
            self.clf = MultinomialNB()
            self.code = 'NB'
        else:
            from sklearn.svm import LinearSVC
            self.clf = LinearSVC(C=1.0, random_state=21)
            self.code = 'SVC'
    
    def get_operation(self):
        operations = ['Nenhuma', 'Troca de palavras', 'Troca de gênero', 'Paráfrase']
        self.operation = st.sidebar.selectbox(label="Operação", options=operations)
    
    @staticmethod
    def get_synonyms(word):
        """
        Get synonyms of a word
        """
        from nltk.corpus import wordnet 

        synonyms = set()
        
        for syn in wordnet.synsets(word, lang='por'): 
            for l in syn.lemmas(lang='por'): 
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklçzxcvbnmáàãâéêíóõôúü'])
                synonyms.add(synonym) 
        
        if word in synonyms:
            synonyms.remove(word)
        
        return list(synonyms)
    
    def synonym_replacement(self, words, stop_words, n):
        if n <= 0: # no word to replace, return the original text
            return words
            
        words = words.split()
        
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            
            if num_replaced >= n: #only replace up to n words
                break

        sentence = ' '.join(new_words)

        return sentence
    
    def apply_operation(self):
        if self.operation == 'Troca de palavras':
            st.write('### Troca de palavras')
            stop_words = self.spacy_nlp.Defaults.stop_words
            percent = st.sidebar.slider("% de palavras trocadas", 0.0, 1.0, value=0.5)
            num_change = int(percent * len(self.original_text))
            self.text = self.synonym_replacement(self.original_text, stop_words, num_change)
            st.write('#**Texto após a troca de palavras**')
            st.write(f'_{self.text}_')

        elif self.operation == 'Troca de gênero':
            return
        
        else:
            from deep_translator import GoogleTranslator

            st.write('### Paráfrase')
            st.write('Alteração da escrita por meio da tradução reversa com a API do Google Tradutor')
            translated = GoogleTranslator(source='pt', target='en').translate(self.original_text)
            back_translated = GoogleTranslator(source='en', target='pt').translate(translated)
            self.text = back_translated
            st.write('**Texto em inglês**')
            st.write(f'_{translated}_')
            st.write('**Texto traduzido de volta ao português**')
            st.write(f'_{self.text}_')

    def load_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.feature_extraction.text import TfidfTransformer

        lem = self.CustomPortugueseLemmatizer(self.spacy_nlp)
        self.pipe = Pipeline([('lemmatizer', lem), 
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', self.clf), 
        ])

    @staticmethod
    def copyright_note():
        st.markdown('----------------------------------------------------')
        st.markdown('Criado por Caio Cedrola Rocha, 2022.')

def main():
    # Create App
    app = App()
        
    # Copyright footnote
    app.copyright_note()

if __name__ == '__main__': main()
