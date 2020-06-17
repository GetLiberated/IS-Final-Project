import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

import tensorflow as tf
sess = tf.Session()
graph = tf.get_default_graph()
from tensorflow.python.keras.backend import set_session
set_session(sess)
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
dataset = json.loads(open('dataset.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the user input - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word and lower - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenize and lemmatize patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

# This function will output a list of intents and the probabilities, their likelihood of matching the correct intent
def predict_class(sentence):
    bag = bag_of_words(sentence, words, show_details=False)
    with graph.as_default():
        set_session(sess)
        # predict user input
        res = model.predict(np.array([bag]))[0]
    # avoid too much overfitting
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, dataset):
    # get predicted user input label
    tag = ints[0]['intent']
    list_of_intents = dataset['data']
    # get the response list from dataset
    for i in list_of_intents:
        if(i['tag']== tag):
            # randomly select response
            result = random.choice(i['responses'])
            break
    return result

def sendResponse(msg):
    ints = predict_class(msg)
    res = getResponse(ints, dataset)
    return jsonify(res)

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
@cross_origin()
def OK():
  return 'OK'


@app.route('/', methods=['POST'])
@cross_origin()
def respond():
  return sendResponse(request.get_json(force=True)['sentence'])

app.run(threaded=True)