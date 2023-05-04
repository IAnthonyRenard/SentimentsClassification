from tensorflow import keras
import pipeline
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



stop_words=set(stopwords.words('english'))
eng_words=[i.lower() for i in words.words()]
lm=WordNetLemmatizer()
eng_words_lem=[lm.lemmatize(i) for i in eng_words]

loaded_model = keras.models.load_model('modeles/model_gru_glove.h5')

text="The film is good"
text_treat=pipeline.process_text(text,rejoin=False,lemm_or_stemm="lem",min_len_words = 3,eng_words=eng_words_lem)
polarity = pipeline.prediction(text_treat)

print( pipeline.decode(polarity))
