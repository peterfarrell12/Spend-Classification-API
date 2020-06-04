import re
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib 
import nltk
import math
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')
np.random.seed(500)




def preprocess(column):
  
  column = [str(i) for i in column]

  column = [entry.lower() for entry in column]

  column = [re.sub(r'[^A-Za-z ]+', '', entry) for entry in column]

  column= [word_tokenize(entry) for entry in column]

  stop_words = set(stopwords.words('english'))

  column = [[w for w in item if not w in stop_words] for item in column]

  column = [[w for w in item if len(w)>2] for item in column]

  column = [[' '.join(w for w in item)] for item in column]

  list_of_strings = []

  for c in column:
    list_of_strings.append(c[0])


  return list_of_strings

#Below Function Trains the Model -> Needs to Be Given PreProcessed Data, and Column Title Joined Cols if L1, Just Description If L2
def train_model1(Corpus, col_title):
    
  Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['joined_cols'],Corpus[col_title],test_size=0.3)
  
  Encoder = LabelEncoder()

  Train_Y = Encoder.fit_transform(Train_Y)
  Test_Y = Encoder.fit_transform(Test_Y)
  
  pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('clf', svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))])
  
  model = pipeline.fit(Train_X, Train_Y)
  
  vectorizer = model.named_steps['vect']
  clf = model.named_steps['clf']
  cats = []
  for label in Encoder.classes_:
    print('Category: ' + label)
    cats.append(label)
  print("Accuracy Score: " + str(model.score(Test_X, Test_Y)))
  return [model, cats]

#Preprocessing Of The Dataset, Doc and Level
def level_dataset(doc, level):
  
  l1_training_set = doc
  pp_Desc = preprocess(l1_training_set['Description'])
  pp_Supp = preprocess(l1_training_set['Supplier'])
  l1_training_set['PP_Description'] = pp_Desc
  l1_training_set['PP_Supplier'] = pp_Supp
  new_l1_training_set = l1_training_set[[level,'PP_Supplier','PP_Description']]
  new_l1_training_set['joined_cols'] = new_l1_training_set['PP_Description'] + ' ' + new_l1_training_set['PP_Supplier']

  return new_l1_training_set

#Seperating Out L1


def getCats(dataset):
  doc = pd.read_excel(dataset)
  levels = []

  for cat in doc['Category']:
    if cat not in levels:
      levels.append(cat)
  return levels


def train(dataset, name):

    doc = pd.read_excel(dataset)
    dataset = level_dataset(doc, 'Category')
    trained_model, cats = train_model1(dataset, 'Category')
    print(cats)
    print(trained_model.classes_)
    print(name)
    joblib.dump(trained_model, f'{name}.model')
    joblib.dump(cats, f'{name}cats')

def check_file(data):
  file = pd.read_excel(data)
  cols = ["Supplier", "Description", "Category"]
  column = list(file.columns)

  if(len(column) != 3):
    print("Length Of Columns < 3")
    return False
  elif(cols != column):
    print("Columns  Not Equal")
    return False
  elif(len(file) < 100):
    print("Not Enough Rows")
    return False
  else:
    return True
