import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Initialize the lemmatizer to reduce words to their base form
lemmatizer = WordNetLemmatizer()

# Load the intents file which contains the dataset for training the chatbot
intents = json.loads(open('data/intents.json').read())

# Initialize lists to hold words, classes, and documents
words = []
classes = []
documents = []

# Define a list of characters to ignore during tokenization
ignore_letters = ['?', '!', ',', '.']

# Process each intent and its associated patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern)
        # Add the tokens to the words list
        words.extend(word_list)
        # Add the tokenized pattern and its associated tag to documents
        documents.append((word_list, intent['tag']))
        # Add the tag to classes if it is not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize the words and remove duplicates, ignoring specified characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes to ensure a consistent order for the output vector
classes = sorted(set(classes))

# Save the words and classes lists to disk for later use
pickle.dump(words, open('models/words.pkl', 'wb'))
pickle.dump(classes, open('models/classes.pkl', 'wb'))

# Initialize the training data and create an empty output array for each class
training = []
output_empty = [0] * len(classes)

# Create the training data by converting patterns into bag of words and associating them with their class tags
for document in documents:
    bag = []
    word_patterns = document[0]
    # Lemmatize and convert each word in the pattern to lowercase
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # Create the bag of words array
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Create the output row where the index of the current tag is 1 and the rest are 0
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    # Add the bag of words and output row to the training set
    training.append([bag, output_row])

# Find the maximum length of the training data for padding
max_length = max(len(x) for x, y in training)

# Pad the training data to ensure all sequences are of equal length
for i, (x, y) in enumerate(training):
    padding_length = max_length - len(x)
    if padding_length > 0:
        x = x + [0] * padding_length
    padding_length = max_length - len(y)
    if padding_length > 0:
        y = y + [0] * padding_length
    training[i] = (x, y)

# Shuffle the training data to ensure random order during training
random.shuffle(training)
training = np.array(training)

# Split the training data into input (X) and output (Y) variables
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Define the neural network model
model = Sequential()
# Add the input layer with 128 neurons and ReLU activation function
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# Add dropout with 0.5 rate to prevent overfitting
model.add(Dropout(0.5))
# Add a hidden layer with 64 neurons and ReLU activation function
model.add(Dense(64, activation='relu'))
# Add dropout with 0.5 rate to prevent overfitting
model.add(Dropout(0.5))
# Add the output layer with softmax activation function for classification
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model using stochastic gradient descent (SGD) optimizer with learning rate decay and momentum
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model on the training data
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model to a file for later use
model.save('models/chatbotmodel.keras', hist)

print('Model training done')
