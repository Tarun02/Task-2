import pickle
import sys
import json
import argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def loading_model():
    loaded_model = pickle.load(open('../artifacts/model.pkl', 'rb'))
    features = pickle.load(open('../artifacts/features.pkl', 'rb'))
    
    return loaded_model, features

def preprocessing_data(path):
    pd_data = pd.read_json(path, lines=True)
    X = pd_data[['review_body']]    
    
    return X
    
def prediction(model, features, data):
    
    count_vec = CountVectorizer(vocabulary=features)
    data_counts = count_vec.transform(data['review_body'])
    
    tfidf_transformer = TfidfTransformer()
    final_processed_data = tfidf_transformer.fit_transform(data_counts)
    
    predictions = model.predict(final_processed_data)
    
    return predictions

def main():
    
    file_path = sys.argv[1]
    print("File path:{}".format(file_path))
    X = preprocessing_data(file_path)
    model, features = loading_model()
    
    predictions = prediction(model, features, X)
    
    output_file = open("predictions/output.txt", "wb")

    output_file.write(predictions)

    output_file.close()

if __name__ == "__main__":
    main()