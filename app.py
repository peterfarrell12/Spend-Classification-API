import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import model1
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from flask_cors import CORS, cross_origin
import pandas as pd




app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
# api = Api(app)

# if not os.path.isfile('clfModel.model'):
#     train()

model = joblib.load('clfModel.model')
# cv = joblib.load('vect')
# routes
@app.route('/predict', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','application/json'])
def predict():
    # get data
        posted_data = request.get_json()
        message = posted_data['message']
        modelName = posted_data['model']

        categories = posted_data['categories']
        # noOfCategories = len(posted_data['categories'])
        model = joblib.load(modelName)
        categories = joblib.load(f"{categories}")
        # categories = list(model.classes_)
        # print(type(categories))

        print(message)
        print(categories)
        print(modelName)
        predictions = []
        # message1 = [message]
        # X = cv.transform(message1)
        for m in message:
            message1 = [m]
            prediction = model.predict(message1)
            for i in range(len(categories)):
                if prediction == i:
                    predicted_class = categories[i]
                    predictions.append(predicted_class)
            # if prediction == [0]:
            #     predicted_class = 'Chemicals'
            # elif prediction == [1]:
            #     predicted_class = 'IT & End-User Equipment'
            # elif prediction == [2]:
            #     predicted_class = 'Packaging'
        return jsonify({
            'Prediction': predictions
        })


@app.route('/test', methods=['POST', 'GET'])
def predictNew():
    f = request.files['file']
    n = request.form['name']
    check = model1.check_file(f)

    if (check):
        try:
             model1.train(f, n)
             model1.getCats(f)
             return f'{n}'
        except:
            return "Failure"
       
    else:
        return "Failure"


if __name__ == '__main__':
    app.run(port = 5000, debug=True)
