#!/usr/bin/env python3
# coding: utf-8
# Config

import streamlit as st
import random
import numpy as np
import spacy

import pickle
import os

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from src.CustomPortugueseLemmatizer import CustomPortugueseLemmatizer

# Program
class App():
    def __init__(self):
        # Show logo, title and description
        self.show_logo()
        st.markdown('''## **NLP Playground**''')
        self.language = self.select_language()
        self.show_description()
        self.get_text()
        self.load_spacy()
        self.get_classifier()
        self.get_operation()
        self.apply_operation()
        self.load_pipeline()
        self.predict_level()

    @staticmethod
    def show_logo():
        st.sidebar.image('logo.png')

    @staticmethod
    def select_language():
        return st.selectbox('Language', ('Português', 'English'))

    def show_description(self):
        st.markdown('----------------------------------------------------')
        if self.language == 'Português':
            st.markdown('''# Como está a minha escrita?
## Um classificador automático do nível de escrita de um texto
Escreva um pequeno texto (0 a 500 palavras) sobre um assunto específico e descubra se seu perfil de escrita se encaixa nos níveis do Ensino Fundamental I, Ensino Fundamental II, Ensino Médio ou Ensino Superior.
Você também pode descobrir como operações de NLP (Natural Language Processing) podem influenciar na classificação, por meio dos controles na barra lateral!
''')
        else:
            st.markdown('''# How is my writing?
## An automatic classifier of the writing level of a text
Write a short text (0 to 500 words) in Portuguese on a specific topic and find out if your writing profile fits the levels of Elementary School I, Elementary School II, High School or Higher Education.
You can also find out how NLP (Natural Language Processing) operations can influence the classification, through the controls in the sidebar!
''')

    def get_text(self):
        if self.language == 'Português':
            text = st.text_area('Escreva um texto e aperte Ctrl+Enter para enviar.')
        else:
            text = st.text_area('Write a text and press Ctrl+Enter to send.')
        self.original_text = text
        self.text = text

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
        if self.language == 'Português':
            operations = ['Nenhuma', 'Troca de palavras', 'Troca de gênero', 'Paráfrase']
            label = 'Operação'
        else:
            operations = ['None', 'Word swap', 'Gender swap', 'Paraphrase']
            label = 'Operation'
        self.operation = st.sidebar.selectbox(label=label, options=operations)

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
    
    def swap_gender(self, text):
        # map of pronouns
        gen_map = {
            'ele': 'ela', 
            'ela': 'ele', 
            'eles': 'elas', 
            'elas': 'eles', 
            'meu': 'minha', 
            'minha': 'meu', 
            'meus': 'minhas', 
            'minhas': 'meus', 
            'teu': 'tua', 
            'tua': 'teu', 
            'teus': 'tuas', 
            'tuas': 'teus', 
            'seu': 'sua', 
            'sua': 'seu', 
            'seus': 'suas', 
            'suas': 'seus', 
            'este': 'esta', 
            'esta': 'este', 
            'estes': 'estas', 
            'estas': 'estes', 
            'esse': 'essa', 
            'essa': 'esse', 
            'esses': 'essas', 
            'essas': 'esses', 
            'aquele': 'aquela', 
            'aquela': 'aquele', 
            'aqueles': 'aquelas', 
            'aquelas': 'aqueles', 
            'àquele': 'àquela', 
            'àquela': 'àquele', 
            'àqueles': 'àquelas', 
            'àquelas': 'àqueles', 
            'mesmo': 'mesma', 
            'mesma': 'mesmo', 
            'mesmos': 'mesmas', 
            'mesmas': 'mesmos', 
            'próprio': 'própria', 
            'própria': 'próprio', 
            'próprios': 'próprias', 
            'próprias': 'próprios', 
            'todo': 'toda', 
            'toda': 'todo', 
            'todos': 'todas', 
            'todas': 'todos', 
            'algum': 'alguma', 
            'alguma': 'algum', 
            'alguns': 'algumas', 
            'algumas': 'alguns', 
            'um': 'uma', 
            'uma': 'um', 
            'uns': 'umas', 
            'umas': 'uns', 
            'certo': 'certa', 
            'certa': 'certo', 
            'certos': 'certas', 
            'certas': 'certos', 
            'vários': 'várias', 
            'várias': 'vários', 
            'muito': 'muita', 
            'muita': 'muito', 
            'muitos': 'muitas', 
            'muitas': 'muitos', 
            'quanto': 'quanta', 
            'quanta': 'quanto', 
            'quantos': 'quantas', 
            'quantas': 'quantos', 
            'tanto': 'tanta', 
            'tanta': 'tanto', 
            'tantos': 'tantas', 
            'tantas': 'tantos', 
            'outro': 'outra', 
            'outra': 'outro', 
            'outros': 'outras', 
            'outras': 'outros', 
        }
        
        word_list = []

        for token in self.spacy_nlp(text):
            word = token.text
            if word in gen_map.keys():
                word = gen_map[word]
            word_list.append(word)
            
        sentence = ' '.join(word_list)
        return sentence

    def apply_operation(self):
        if not self.original_text:
            return
        st.write(f'### {self.operation}')
        if self.operation == 'Troca de palavras' or self.operation == 'Word swap':
            stop_words = self.spacy_nlp.Defaults.stop_words
            if self.language == 'Português':
                slider_label = '% de palavras trocadas'
                label = 'Texto após a troca de palavras'
            else:
                slider_label = '% of words that will be swapped'
                label = 'Text after swapping words'
            percent = st.sidebar.slider(label, 0.0, 1.0, value=0.5)
            num_change = int(percent * len(self.original_text))
            self.text = self.synonym_replacement(self.original_text, stop_words, num_change)
            st.write(f'**{label}**')
            st.write(f'_{self.text}_')
        elif self.operation == 'Troca de gênero' or self.operation == 'Gender swap':
            if self.language == 'Português':
                st.warning('No momento, somente são trocados pronomes!')
                label = 'Texto após a troca de gênero'
            else:
                st.warning('At the moment, only pronouns are exchanged!')
                label = 'Text after swapping gender'
            self.text = self.swap_gender(self.original_text)
            st.write(f'**{label}**')
            st.write(f'_{self.text}_')
        
        elif self.operation == 'Paráfrase':
            from deep_translator import GoogleTranslator
            
            if self.language == 'Português':
                st.write('Alteração da escrita por meio da tradução reversa com a API do Google Tradutor')
                label1 = 'Texto em inglês'
                label2 = 'Texto traduzido de volta ao português'
            else:
                st.write('Change of writing through reverse translation with Google Translate API')
                label1 = 'Text in English'
                label2 = 'Text translated back into Portuguese'
            translated = GoogleTranslator(source='pt', target='en').translate(self.original_text)
            back_translated = GoogleTranslator(source='en', target='pt').translate(translated)
            self.text = back_translated
            st.write(f'**{label1}**')
            st.write(f'_{translated}_')
            st.write(f'**{label2}**')
            st.write(f'_{self.text}_')
        else:
            if self.language == 'Português':
                st.write('**Texto escrito**')
            else:
                st.write('**Written text**')
            st.write(f'_{self.text}_')

    def load_pipeline(self):
        path = 'pickle'
        if os.path.isdir(path):
            # Read the classifier from pickle
            with open(f'{path}/pipeline_{self.code}.pickle', 'rb') as file:
                self.pipe = pickle.load(file)

    def predict_level(self):
        if not self.text:
            return
        X_test = [self.text]
        y_pred = self.pipe.predict(X_test)
        predicted_level = int(y_pred[0])
        if self.language == 'Português':
            st.write('### Seu nível de escrita classificado é: ')
            if predicted_level == 1:
                st.write('Ensino Fundamental I')
            elif predicted_level == 2:
                st.write('Ensino Fundamental II')
            elif predicted_level == 3:
                st.write('Ensino Médio')
            else:
                st.write('Ensino Superior')
        else:
            st.write('### Your graded writing level is: ')
            if predicted_level == 1:
                st.write('Elementary School I')
            elif predicted_level == 2:
                st.write('Elementary School II')
            elif predicted_level == 3:
                st.write('High School')
            else:
                st.write('Higher Education')

    def copyright_note(self):
        st.markdown('----------------------------------------------------')
        if self.language == 'Português':
            st.markdown('Criado por Caio Cedrola Rocha, 2022.')
        else:
            st.markdown('Created by Caio Cedrola Rocha, 2022.')

def main():
    # Create App
    app = App()
        
    # Copyright footnote
    app.copyright_note()

if __name__ == '__main__': main()
