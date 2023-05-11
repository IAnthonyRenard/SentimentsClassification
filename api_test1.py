import json
import requests
import pytest

def reponse_test_api(url):
    # Données à prédire
    data = {
        'message': 'The film is good'
    }

    # Convertir les données en JSON
    #data_json = json.dumps(data)

    # Envoyer la requête POST à l'API Flask
    response = requests.post(url, data=data)

    # Afficher la réponse de l'API Flask
    #prediction = response.json()[0]

    # Afficher la réponse de l'API Flask
    print(response.text)
    print(response.status_code)
    
    return response.text
    
#test_api("http://127.0.0.1:3001/predict_post") #Test en local
#test_api("https://sentimentsclassification.herokuapp.com/predict_post")# Tester une requête avec le message enregistré ci-dessus
#test_api("https://sentimentsclassification.herokuapp.com") # Tester une requête avec le message enregistré ci-dessus

# Mise en place du test Pytest
def test_modele_post():
    assert reponse_test_api("https://sentimentsclassification.herokuapp.com/predict_post")=="Positif" 