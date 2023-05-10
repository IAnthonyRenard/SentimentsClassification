from flask import Flask, render_template, request, redirect, session
import pipeline
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


stop_words=set(stopwords.words('english'))
eng_words=[i.lower() for i in words.words()]
lm=WordNetLemmatizer()
eng_words_lem=[lm.lemmatize(i) for i in eng_words]
 
app = Flask(__name__)
 

#Affichage de IHM via message.html
@app.route('/')
def view_form():
    return render_template('message.html')
 
#utilisation de la méthode GET
@app.route('/predict_get', methods=['GET'])
def predict_get():
    if request.method == 'GET':        
        text = request.args['message']
        text_treat=pipeline.process_text(text,rejoin=False,lemm_or_stemm="lem",min_len_words = 3,eng_words=eng_words_lem)
        polarity = pipeline.prediction(text_treat)
    
    return pipeline.decode(polarity)
 
#utilisation de la méthode POST
@app.route('/predict_post', methods=['POST'])
def predict_post():
    if request.method == 'POST':
        
        text = request.form['message']
        text_treat=pipeline.process_text(text,rejoin=False,lemm_or_stemm="lem",min_len_words = 3,eng_words=eng_words_lem)
        polarity = pipeline.prediction(text_treat)
    
    return pipeline.decode(polarity)
        
 
if __name__ == '__main__':
    app.run(https://sentimentsclassification.herokuapp.com)
    
    #port=3001)