# Objectif du projet

Fournir un prototype d’un produit IA permettant de prédire le sentiment associé à un Tweet (positif ou négatif).
Les Tweets sont disponibles sous : https://www.kaggle.com/datasets/kazanova/sentiment140

# Opérations sur les entrées

Step 1 : Exploratory data analysis (EDA) : nettoyage et état des lieux des données disponibles
Step 2 : Prise en compte d'un échantillon équilibré parmi les 1.6 millions de Tweets disponibles
Step 3 : Traitement du texte --> 2 choix possibles : Tweets finaux lemmatisés ou racinisés (les 2 options sont disponibles)

# Création des modèles
 
## 3 modèles "simples" avec 2 embeddings différents 

eXtreme Gradient Boosting (XGB), Multinomial Naïve Bayse (NMB) et Logistic Regression (LR) sont les 3 modèles choisis. 
Pour ces modèles, 2 types de plongements de mots (embeddings) ont été utilisés : Bag-of-Words et TF-IDF.

## 3 modèles "complexes" avec 4 embeddings différents 
Simple Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM) et Gated Recurrent Unit (GRU) sont les 3 modèles qui ont été choisis pour ce travail.
Pour ces modèles, 4 types de plongements de mots (embeddings) ont été utilisés : Word2Vec, GloVe, FastText, et USE.

La différences entre tous ces modèles et les embeddings associés sont disponibles sous la page Web suivante (disponible sous GitHub) : 
Renard_Anthony_3_article_de_blog_042023

## 1 modèle de BERT
Modèle très performant grâce notamment à sa méthode MLM et l'utilisation de masque d'attention


# Mise en place de tests unitaires
Utilisation de Pytest pour la mise en place de test unitaires sur la partie EDA et création des modèles. 

# Enregistrement des modèles
Enregistrement des modèles (artifacts) et de leurs performances sur MLFLOW

# API
Création d'une API FLASK pour l'interrogation du meilleur modèle sur un serveur externe (Heroku)

# Déploiement 

Step 1: versionning sur GitHub 
Step 2 : Mise en place de GitHubActions pour l'interrogation du modèle via l'API
Step 3 : déploiement du modèle sur le serveur externe (Heroku)


Note : à chaque PUSH sur GitHub, le serveur en ligne est automatiquement mis à jour.

# Découpage des livrables disponibles sous GitHub : https://github.com/IAnthonyRenard/SentimentsClassification

-Renard_Anthony_1_modele_042023 : contient les fichiers "API" pour interroger le modèle sur le serveur Cloud Heroku<br/> 
-Renard_Anthony_2_scripts_042023 : contient toutes les opérations sur les entrées et les algorithmes de création des modèles cités précédemment<br/> 
-Renard_Anthony_3_article_de_blog_042023 : contient une présentation et une comparaison des trois approches (“Modèle sur mesure simple” et “Modèle sur mesure avancé”, “Modèle avancé BERT”) ainsi que l'explication de la démarche orientée MLOps mise en œuvre<br/> 
-Renard_Anthony_4_presentation_042023 : présentation commenté visuelle du projet<br/>  
-tests/ : contient tous les fichiers permettant de lancer les tests PyTest <br> 

# FIN