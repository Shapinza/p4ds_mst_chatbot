from data_process import data_process
from model_train import model_train
from pyexpat import model
from bot import *
from config import load_config
import telebot
import random
import numpy
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


# change philosopher here
philosopher_name = "Marx"
config = load_config("config.yaml")
bot_token = config["telegram"]["token"]
chat_id = config["telegram"]["chat_id"]
bot = telebot.TeleBot("5730786911:AAGk0UB0zUD3DRFyKL_ZHyI9mVbUkFqzc1A")


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


model = model_train(training, output, philosopher_name)


@bot.message_handler()
def echo_all(inp):
    results = model.predict([bag_of_words(inp.text, words)])

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

    bot.reply_to(inp, msg)


bot.infinity_polling()
