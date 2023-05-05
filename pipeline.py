import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import  PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import RegexpTokenizer
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
import mlflow
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
stop_words=set(stopwords.words('english'))
from tensorflow import keras




# Chargement du modèle
#logged_model = 'C:/Users/Utilisateur/PROJET7/mlruns/599463961880450291/4714c7182bcc40a2b29bb3f5afc0248b/artifacts/model'
#loaded_model = mlflow.pyfunc.load_model(logged_model)

#Charger le h5
#loaded_model = keras.models.load_model('C:/Users/Utilisateur/PROJET7/modeles/model_gru_glove.h5')
loaded_model = keras.models.load_model('modeles/model_gru_glove.h5')

#Chargement du tokeniser
def token(text):
    # number of vocab to keep
    max_vocab = 18000
    # length of sequence that will generate
    max_len = 15
    
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    text_token = pad_sequences(sequences, maxlen=max_len, padding="post")
    
    return text_token





def process_text(doc,
                   rejoin=False,
                   lemm_or_stemm=None,
                   freq_rare_words=None,
                   min_len_words=None,
                   force_is_alpha=True,
                   eng_words=None,
                   extra_words=None) :
    
    # list unique words
    #if not list_rare_words:
        #list_rare_words=[] #Si non on crée liste vide
        
    # forcer le document à etre en minuscule
    doc=doc.lower().strip() # Mise en minuscule et suppression des espaces avant et apres la chaine de caractère
    #print("-----> Passage en minuscule effectué") 
    
    # tokenize
    tokenizer = RegexpTokenizer(r"\w+")
    raw_token_list=tokenizer.tokenize(doc)
    #print("-----> Passage en token effectué") 
    
    # stopwords : suppresion des stop_words de la liste
    cleaned_tokens_list = [w for w in  raw_token_list if w not in stop_words]
    #print("-----> Suppression des stopwords effectué") 
    
    # création de la liste des mots rares
    tmp=pd.Series(cleaned_tokens_list).value_counts()
    list_rare_words=tmp[tmp==freq_rare_words]
    list_rare_words=list(list_rare_words.index) #transformation du vecteur en une liste
    #print("-----> Liste des mots rares qui apparaissent", freq_rare_words, "fois effectuée. Il y a", len(list_rare_words) ,"mots différents")
    
    
   #no rare token : suppression des tokens appartenant à la liste des mots rares
    non_rare_tokens = [w for w in  cleaned_tokens_list if w not in list_rare_words]
    #print("-----> Suppression des mots rares effectué")
    
    # no more len words : selection des token ayant une longueur min
    more_than_N= [w for w in non_rare_tokens if len(w) >= min_len_words]
    #print("-----> Suppression des mots ayant moins de", min_len_words,"lettres effectué")
    
    #On garde ou non les tokens 100% alphabétiques
    if force_is_alpha :
        alpha_tokens= [w for w in more_than_N if w.isalpha()]
        #print("-----> Conservation des mots 100% alphabétiques effectué")
    else : 
            alpha_tokens =  more_than_N
        
        
    #lem or stem
    if lemm_or_stemm=="lem":
        trans=WordNetLemmatizer()
        trans_text=[trans.lemmatize(i) for i in alpha_tokens]
        #print("-----> Lemmatisation effectué")
    else :
        trans=PorterStemmer()
        trans_text=[trans.stem(i) for i in alpha_tokens]
        #print("-----> Racinisation effectué")
        
    # in english
    if eng_words :
        engl_text=[i for i in trans_text if i in eng_words]
        #print("-----> Conservation des mots anglais effectué")
    else :
        engl_text=trans_text
        
    # Suppresion des mots communs entre les 2 catégories (extra words)
    if extra_words :
        final = [w for w in  engl_text if w not in extra_words] 
    else :
        final=trans_text
        
    #renvoi d'une liste de token ou une chaine de caractère
      
    if rejoin :
        return " ".join(final)
    
    
    #print("-> Corpus ready <-")
    return  final#, list_rare_words




def prediction(text_treat, model=loaded_model):
    tokenised=token(text_treat)
    predictions = model.predict(tokenised)
    predictions = predictions.argmax(axis=1)
    
    return predictions
    
def decode(predictions):
    
    predictions=predictions.tolist()
    if 0 in predictions:
        return "Négatif"
    return "Positif"
    
   
 
    #return predictions
    
    #if predictions == 1:
         #return "Positif"
     #return "Négatif"
    
    
    