import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

# Initialize the lemmatizer to reduce words to their base form
lemmatizer = WordNetLemmatizer()

# Load the intents file which contains the dataset for the chatbot
intents = json.loads(open('data/intents.json').read())

# Load the pre-processed words and classes lists from disk
words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))

# Load the trained model from disk
model = load_model('models/chatbotmodel.keras')

# Function to clean up and tokenize sentences
def clean_up_sentence(sentence):
    # Tokenize the sentence into words
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize each word to its base form
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to convert sentences into a bag of words
def bag_of_words(sentence):
    # Clean up the sentence
    sentence_words = clean_up_sentence(sentence)
    # Initialize a bag of words with zeros
    bag = [0] * len(words)
    # Mark the presence of words in the sentence
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class of a given sentence
def predict_class(sentence):
    # Convert the sentence into a bag of words
    bow = bag_of_words(sentence)
    # Predict the class probabilities using the trained model
    res = model.predict(np.array([bow]))[0]
    # Define an error threshold to filter out low probability classes
    ERROR_THRESHOLD = 0.25
    # Filter and sort the results by probability
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    # Prepare the list of predicted classes with their probabilities
    return_list = []
    for r in results:
        print('r: ', r)
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Function to get a response based on the predicted class
def get_response(intents_list):
    # Get the tag of the first predicted intent
    tag = intents_list[0]['intent']
    # Get the list of intents from the intents file
    list_of_intents = intents['intents']
    # Find and return a random response for the matching intent
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Function to process a user message and generate a response
def chat(message):
    # Predict the class of the user message
    intents_list = predict_class(message)
    # Get a response based on the predicted class
    result = get_response(intents_list)
    return result
