# https://towardsdatascience.com/how-to-create-a-chatbot-with-python-deep-learning-in-less-than-an-hour-56a063bdfc44
# Step 1.  Import libraries and load the data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

words = []
labels = []
documents = []
ignore_letters = ['!', '?', ',', '.']
dataset = json.loads(open('dataset.json').read())

# Step 2. Processing the data
for data in dataset['data']:
    for pattern in data['patterns']:
        # tokenize each word in 'patterns'
        tokenized_word = nltk.word_tokenize(pattern)
        # Tokenize example:
        # "Hi there" -> ["Hi", "there"]

        # normalize the tokenized words by using Lemmatization and lower. Then save it to words
        lemmatized_word = [lemmatizer.lemmatize(word.lower()) for word in tokenized_word]
        words.extend(lemmatized_word)
        # append it to documents including the label for identification
        documents.append((lemmatized_word, data['tag']))
        # Lemmatization and lower the tokenized words example:
        # ["The", "person's", "jackets", "are", "different", "colors"] -> ["the", "person", "jacket", "be", "differ", "color"]

        # Stemming vs Lemmatization
        # ‘Caring’ -> Lemmatization -> ‘Care’ (Stem from dictionary database) ✅
        # ‘Caring’ -> Stemming -> ‘Car’ (Cut off -ing) ❌

        # append label to labels
        if data['tag'] not in labels:
            labels.append(data['tag'])

# sort and remove duplicate word if any
words = sorted(list(set(words)))

# sort and remove duplicate label if any
labels = sorted(list(set(labels)))

# save it, to use in GUI
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(labels,open('classes.pkl','wb'))

# Step 3. Create training and testing data
# training data will be stored here
training = []
# create a template array that consist of 0's to identify which label the words is
label_index_template = [0] * len(labels)

for doc in documents:
    # we need to use bag of words since neural network only accept numbers
    bag = []
    # the document tokenized and lemmatized 'pattern' words
    pattern_words = doc[0]
    # create bag of words array by appending 1 if a word in pattern_words exist in words. Else append 0
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    # specify the words label by assigning 1 to the corresponding label index
    label_index = list(label_index_template)
    label_index[labels.index(doc[1])] = 1
    
    training.append([bag, label_index])

# shuffle training data and turn it into np.array because model only accept np.array
random.shuffle(training)
training = np.array(training)

# create train and test data. X = patterns, Y = labels
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# Step 4. Training the model 
# Create model - 4 layers
model = Sequential()
# Input layer neurons equal to number of patterns to predict and first hidden layer 128 neurons
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
# Second hidden layer 64 neurons
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# Output layer neurons equal to number of labels to predict
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent (SGD) with Nesterov accelerated gradient is more efficient than normal gradient descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train and save the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("Model created")
