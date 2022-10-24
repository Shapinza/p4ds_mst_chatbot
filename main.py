from data_process import data_process
from model_train import model_train
from pyexpat import model
import random
import numpy
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


# change philosopher here
philosopher_name = "Marx"


[words, labels, training, output, data] = data_process(
    "intent.json", philosopher_name)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat(philosopher_name):
    model = model_train(training, output, philosopher_name)
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])

        # to disable debugging, comment out the following few lines
        temp = []
        for a, b in zip(results[0], labels):
            temp.append(b + ": " + str(a))
        print(temp)

        # fallback logic
        if max(results[0]) < 0.75:
            # print no winner, uncomment if not needed
            print("No winner category, use fallback logic")
            print(philosopher_name + ": " +
                  "Sorry I don't understand what you are saying. Could you please try again?")

        else:
            results_index = numpy.argmax(results)
            tag = labels[results_index]

            # print winner category, uncomment if not needed
            print("Winner tag: " + tag, "| Score: " + str(max(results[0])))

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(philosopher_name + ": " + random.choice(responses))

        print("")


chat(philosopher_name)
