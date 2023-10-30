import os
import json
import pickle
import numpy as np
import implicit
import scipy
import joblib


class cfRecommender():

    # Recommender les meilleurs 5 articles utilisant le collaborative filtering

    def __init__(self, sparse_item_user, sparse_user_item):
        self.sparse_item_user = sparse_item_user
        self.sparse_user_item = sparse_user_item

    def implicit_recommendation(self, user_id):
        # Initialize the als model and fit it using the sparse item-user matrix
        model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

        # Calculate the confidence by multiplying it by our alpha value.
        #Initiale
        alpha_val = 15#alpha =40 selon un certain travai à vérifier si c'est réellement bon
        data_conf = (self.sparse_item_user * alpha_val).astype('double')

        #Fit the model
        model.fit(data_conf)
 
        """CREATE USER RECOMMENDATIONS"""
        
        # Use the implicit recommender.
        print ("start recommendation dans implicit recommendations")
        recommended = model.recommend(user_id, self.sparse_user_item)
        implicit_articles = []

        # Get artist names from ids
        for item in recommended:
            idx, score = item
            implicit_articles.append(int(idx)) # cast to int necessary to make it work on json
        
        # to remove duplicated from implicit_articles 
        result = [] 
        [result.append(int(x)) for x in implicit_articles if x not in result]
        return result[:5] #implicit_articles[:5] #result[:5]




def initialize_model():

    #-----Load sparse_user_item matrix
    with open("sparse_user_item.pkl", "rb") as f:
            loaded_sparse_user_item = pickle.load(f)
    print("loaded_sparse_user_item done")

    #-----Load sparse_item_user matrix
    with open("sparse_item_user.pkl", "rb") as f:
            loaded_sparse_item_user = pickle.load(f)
    print("loaded_sparse_item_user done")

    #-----Load best model CF
    #loaded_model = joblib.load("CF_model.joblib")
    #loaded_model = pickle.load(open('CF_model.pkl', 'rb'))
    #with open("CF_model.pkl", "rb") as f:
    #       loaded_model = pickle.load(f)
    #print("loaded_model done")

    cf_object = cfRecommender(loaded_sparse_item_user, loaded_sparse_user_item)
    
    return cf_object



def predict_reco_CF(userId):
    print("Start predict.py, userId ", userId, type(userId)), 
    model = initialize_model()
    return  model.implicit_recommendation(userId)

