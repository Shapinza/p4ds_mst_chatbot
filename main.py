from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy
from uvicorn import run
import os
from data_process import data_process
from model_train import model_train
import nltk
from nltk.stem.lancaster import LancasterStemmer
import random
stemmer = LancasterStemmer()


app = FastAPI()


[words, labels, training, output, data] = data_process(
    "intent.json")

model = model_train(training, output, "Marx")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Food Vision API!"}


@app.post("/net/chat/prediction/")
async def get_response_prediction(sentence):
    if not sentence:
        return {"message": "No sentence provided"}

    results = model.predict([bag_of_words(sentence.text, words)])

    fallbacks = ["I don’t understand your Gen Z lingo. After all, I am a little over 200 years old. Can you say it another way?",
                 "That’s up to you to say, because I can’t give you a reply to that. Tell me something else.",
                 "Some things do not deserve to be graced with a reply. Kidding, I just don’t understand what you are saying. Try again.",
                 "I don’t understand the manner you are speaking, comrade. Don’t submit to incoherence. Tell me something else."]

    # fallback logic
    if max(results[0]) < 0.75:
        msg = random.choice(fallbacks)

    else:
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        msg = random.choice(responses)

    return {
        "response": msg,
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)
