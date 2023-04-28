import pipeline2
from flask import Flask, jsonify, request
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
import numpy as np
import tensorflow as tf


stop_words=set(stopwords.words('english'))
eng_words=[i.lower() for i in words.words()]
lm=WordNetLemmatizer()
eng_words_lem=[lm.lemmatize(i) for i in eng_words]


app = Flask(__name__)

@app.route("/")
def hello():
    return "Le serveur flask fonctionne"

@app.route("/predict2", methods=['GET'])
def get_prediction():
    text = request.args.get('text')
    text_treat=pipeline2.process_text(text,rejoin=False,lemm_or_stemm="lem",min_len_words = 3,eng_words=eng_words_lem)
    input_ids, attention_masks=pipeline2.encode(text_treat,)
    polarity = pipeline2.predict(input_ids, attention_masks,)
    #polarity=json.dumps(polarity, default=str)
    #return jsonify(polarity=polarity)
    
    #return pipeline2.decode(polarity[0]) #Ok fonctionne pour le 1er mot
    polarity=polarity.tolist()
    return pipeline2.decode(polarity)

if __name__ == "__main__":
    app.run(port=3000,debug=True)