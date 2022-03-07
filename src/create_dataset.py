#!/usr/bin/python
# coding: utf-8

# Code adapted from https://towardsdatascience.com/cleaning-preprocessing-text-data-by-building-nlp-pipeline-853148add68a
import os
import pandas as pd

def remove_newlines_tabs(text):
    """
    This function will remove all the occurrences of newlines, tabs, and combinations like: \\n, \\.
    
    arguments:
        input_text: "text" of type "String". 
                    
    return:
        value: "text" after removal of newlines, tabs, \\n, \\ characters.
        
    Example:
    Input : This is her \\ first day at this place.\n Please,\t Be nice to her.\\n
    Output : This is her first day at this place. Please, Be nice to her. 
    
    """
    
    # Replacing all the occurrences of \n,\\n,\t,\\ with a space.
    Formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
    return Formatted_text

def strip_html_tags(text):
    """ 
    This function will remove all the occurrences of html tags from the text.
    
    arguments:
        input_text: "text" of type "String". 
    
    return:
        value: "text" after removal of html tags.
        
    Example:
    Input : This is a nice place to live. <IMG>
    Output : This is a nice place to live.  
    """
    from bs4 import BeautifulSoup

    # Initiating BeautifulSoup object soup.
    soup = BeautifulSoup(text, "html.parser")
    # Get all the text other than html tags.
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_links(text):
    """
    This function will remove all the occurrences of links.
    
    arguments:
        input_text: "text" of type "String". 
                    
    return:
        value: "text" after removal of all types of links.
        
    Example:
    Input : To know more about this website: kajalyadav.com  visit: https://kajalyadav.com//Blogs
    Output : To know more about this website: visit:     
    
    """
    import re

    # Removing all the occurrences of links that starts with https
    remove_https = re.sub(r'http\S+', '', text)
    # Remove all the occurrences of text that ends with .com
    remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
    return remove_com

def remove_whitespace(text):
    """ This function will remove 
        extra whitespaces from the text
    arguments:
        input_text: "text" of type "String". 
                    
    return:
        value: "text" after extra whitespaces removed .
        
    Example:
    Input : How   are   you   doing   ?
    Output : How are you doing ?     
        
    """
    import re

    # Remove leading whitespaces, newline and tab characters
    text = text.lstrip()
    # Create pattern
    pattern = re.compile(r'\s+') 
    Without_whitespace = re.sub(pattern, ' ', text)
    # There are some instances where there is no space after '?' & ')', 
    # So I am replacing these with one space so that It will not consider two words as one token.
    text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
    return text

def remove_non_printable(text):
    """ This function will remove 
        non-printable characters from the text
    arguments:
        input_text: "text" of type "String". 
                    
    return:
        value: "text" after extra whitespaces removed .
        
    Example:
    Input : \ufeff I'm great!
    Output : I'm great!
    
    """
    import re

    # Remove non-printable characters, keeping white spaces and punctuation marks
    text = re.sub('[^\w\s"\'!@#$%&*()-_+=\[\]{}:;.,|]+','', text)
    return text

# Code for text lowercasing
def lower_casing_text(text):
    
    """
    The function will convert text into lower case.
    
    arguments:
         input_text: "text" of type "String".
         
    return:
         value: text in lowercase
         
    Example:
    Input : The World is Full of Surprises!
    Output : the world is full of surprises!
    
    """
    # Convert text to lower case
    # lower() - It converts all upperase letter of given string to lowercase.
    text = text.lower()
    return text

def cleaning(text):
    """
    Do some cleaning in the original text, removing line breaks, tabs, 
    html tags, links, and whitespaces.
    """
    string = remove_newlines_tabs(text)
    string = strip_html_tags(string)
    string = remove_links(string)
    string = remove_non_printable(string)
    string = remove_whitespace(string)
    return string

def preprocessing(text):
    """
    Pre-process the dataset so that it can be used for NLP.
    Python differentiates lower case and upper case, so it is
    necessary to transform all the text to lower case, so that
    "The" and "the" are counted as the same word in NLP. 
    """
    string = lower_casing_text(text)
    return string

def main():
    # Parse through directories and read the files
    texts = []
    for root, directories, files in os.walk('dataset_caed'):
        for d in directories:
            for _, _, files2 in os.walk(f'dataset_caed/{d}'):
                for f in files2:
                    with open(f'dataset_caed/{d}/{f}') as text_file:
                        text = ''
                        for line in text_file.readlines():
                            text_clean = cleaning(line)
                            text_clean = preprocessing(text_clean)
                            text += text_clean
                        texts.append({'id': f.rstrip('.txt'), 'text': text, 'label': d[0]})
    
    df = pd.DataFrame(texts)
    df.to_csv('dataset.csv', index=False)

if __name__ == '__main__': main()