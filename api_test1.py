import json
import requests

def test_api(url):
    # Données à prédire
    data = {
        'message': 'the film is good'
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
    
    
    
#test_api("http://127.0.0.1:3001/predict_post")

test_api("http://0.0.0.0/predict_post")

#test_api("https://sentimentsclassification.herokuapp.com/predict_post")
