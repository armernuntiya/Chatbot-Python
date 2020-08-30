import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        word, labels, training, output = pickle.load(f)

except : 
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intents["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            word.extend(wrds)
            docs_x.append(pattern)
            docs_y.append(intent['tag'])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(doc_x):
        bag=[]
        wrds=[stemmer.stem(w) for w in doc]

        for w in word :
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)
    

    training = numpy.array(training)
    output = np.array(output)

    with open("data.pickle","rb") as f:
        pickle.dump((word, labels, training, output),f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0], activation="softmax"))
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training,output,n_epoch=1000,batch)
    modal.save("model.tflearn")

def bag_of_word(s,words):
    bag = [0 for _ in range(len(word))]

    s_word = nltk.word_takenize(s)
    s_word = [stemmer.stem(word.lower()) for word in s_word]

    for se in s_words:
        for i, w in eumerate(words):
            if w == se:
                bag[i]=1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("YOU: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_word(inp,words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I did't get that, try again.")

chat()