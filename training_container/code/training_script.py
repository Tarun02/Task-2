import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def reading_data():
    initial_train_data = pd.read_json("data/dataset_en_train.json", lines=True)
    
    return initial_train_data
    
def preprocess_data(data):
    data = data[['review_body','product_category']]
    
    # Remove the product_category as it is the target item.
    X = data.drop(['product_category'], axis=1)
    y = data['product_category']
    
    return X,y
    
def training_model(X, y):
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(X['review_body'])
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    model = MultinomialNB().fit(X_tfidf, y)
    
    return model, count_vect

def saving_model(model, vectors):
    file = open("output/model.pkl","wb")
    pickle.dump(model, file)
    
    features_file = open("output/features.pkl", "wb")
    pickle.dump(vectors.vocabulary_, features_file)

def main():
    read_data = reading_data()
    X, y = preprocess_data(read_data)
    final_model, features = training_model(X, y)
    
    saving_model(final_model, features)

if __name__ == "__main__":
    main()