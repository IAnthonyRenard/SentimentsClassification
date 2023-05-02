import ipytest
import nbimporter
import os
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
from nltk.stem import  PorterStemmer




os.chdir("C:/Users/Utilisateur/PROJET7") # Je vais chercher les fonctions dans le fichier placé dans PROJET7
from Renard_Anthony_2_scripts_042023 import process_text

os.chdir("C:/Users/Utilisateur/PROJET7/tests") #Je me replace dans le dossier où est situés mes tests


def test_should_process_text_lem():
    
    stop_words=set(stopwords.words('english'))
    
    eng_words=[i.lower() for i in words.words()]
    lm=WordNetLemmatizer()
    eng_words_lem=[lm.lemmatize(i) for i in eng_words]
    
    
    assert process_text('Pytest is testing if sentences are treated', rejoin=True, lemm_or_stemm="lem",min_len_words = 3,eng_words=eng_words_lem,stop_words=stop_words)=="pytest testing sentence treated"
    
    
def test_should_process_text_stem():
    stop_words=set(stopwords.words('english'))
    
    eng_words=[i.lower() for i in words.words()]
    ps=PorterStemmer()
    eng_words_stem=[ps.stem(i) for i in eng_words]
    
    
    assert process_text('Pytest is testing if sentences are treated', rejoin=True, lemm_or_stemm="stem",min_len_words = 3,eng_words=eng_words_stem,stop_words=stop_words)=="pytest test sentenc treat"