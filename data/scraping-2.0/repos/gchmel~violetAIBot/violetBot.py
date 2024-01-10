import math
from datetime import datetime
import time
from abc import ABCMeta, abstractmethod

import random
import json
import pickle
import os
import openai

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from nltk.stem import WordNetLemmatizer

import smartHistory
from mood import Mood

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
import numpy as np
from tensorflow.python.keras.models import load_model
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

OPEN_AI_API_TOKEN = json.loads(open("./sources/settings.json").read())['OPEN_AI_API_TOKEN']


class IAssistant(metaclass=ABCMeta):

    @abstractmethod
    def train_model(self):
        """ Implemented in child class """

    @abstractmethod
    def request_tag(self, message):
        """ Implemented in child class """

    @abstractmethod
    def get_tag_by_id(self, id):
        """ Implemented in child class """

    @abstractmethod
    def request_method(self, message):
        """ Implemented in child class """

    @abstractmethod
    def request(self, message):
        """ Implemented in child class """


class VioletBot(IAssistant):

    def __init__(self, intents, intent_methods={}, model_name="assistant_model", history_size=10):
        self.intents = intents
        self.intents_file = intents
        self.intent_methods = intent_methods
        self.model_name = model_name
        self.mood = dict()
        self.original_message = ""
        self.history = smartHistory.SmartHistory(history_size, model_name)
        self.sia = SentimentIntensityAnalyzer()

        if intents.endswith(".json"):
            self.load_json_intents(intents)

        self.lemmatizer = WordNetLemmatizer()

    def load_json_intents(self, intents):
        self.intents = json.loads(open(intents).read())

    def get_feelings(self, person):
        return self.mood.get(person).get_mood()

    def train_model(self):
        self.load_json_intents(self.intents_file)
        self.words = []
        self.classes = []
        documents = []
        ignore_letters = []

        intents_length = len(self.intents)
        for i, intent in enumerate(self.intents):
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                self.words.extend(word)
                documents.append((word, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
            #print(f'[DEBUG]: Broke down into words {i} intent out of {intents_length}')

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(list(set(self.words)))

        self.classes = sorted(list(set(self.classes)))

        training = []
        output_empty = [0] * len(self.classes)

        for doc in documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

    def save_model(self, model_name=None):
        if model_name is None:
            self.model.save(f"./sources/{self.model_name}/model.h5", self.hist)
            pickle.dump(self.words, open(f'./sources/{self.model_name}/words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'./sources/{self.model_name}/classes.pkl', 'wb'))
        else:
            self.model.save(f"./sources/{model_name}/model.h5", self.hist)
            pickle.dump(self.words, open(f'./sources/{model_name}/words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'./sources/{model_name}/classes.pkl', 'wb'))

    def load_model(self, model_name=None):
        if model_name is None:
            self.words = pickle.load(open(f'./sources/{self.model_name}/words.pkl', 'rb'))
            self.classes = pickle.load(open(f'./sources/{self.model_name}/classes.pkl', 'rb'))
            self.model = load_model(f'./sources/{self.model_name}/model.h5')
        else:
            self.words = pickle.load(open(f'./sources/{model_name}/words.pkl', 'rb'))
            self.classes = pickle.load(open(f'./sources/{model_name}/classes.pkl', 'rb'))
            self.model = load_model(f'./sources/{model_name}/model.h5')

    def _clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def _bag_of_words(self, sentence, words):
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)

    def _predict_class(self, sentence):
        self.original_message = sentence

        p = self._bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.8
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def _get_response(self, author, ints, intents_json, message):
        try:
            tag = ints[0]['intent']
            if tag == "i_dont_understand":
                return self._learn_from_new_type(intents_json, message)
            list_of_intents = intents_json
            for i in list_of_intents:
                if i['tag'] == tag:
                    if self.mood.get(author) is None:
                        self.mood[author] = Mood(self.model_name, author)
                    mood_change = self._get_mood_change(message)
                    self.mood.get(author).update_mood(mood_change[0], mood_change[1])
                    mood = self.mood.get(author).get_mood()
                    choice = mood + "_responses"
                    result = random.choice(i[choice])
                    break
            self.learn_new_input(message, tag)
        except IndexError:
            result = "I don't understand!"
        self.history.put(('bot', result))
        return result

    def learn_new_tag(self, intents_json, message, tag, answer):
        new_entry = {
            "tag": tag,
            "patterns": [
                message
            ],
            "neutral_responses": [answer],
            "sad_responses": [answer],
            "happy_responses": [answer],
            "depressed_responses": [answer],
            "angry_responses": [answer],
            "love_responses": [answer],
            "best_friends_responses": [answer],
        }
        intents_json.append(new_entry)
        json.dump(intents_json, open(f'./sources/{self.model_name}/intents.json', 'r+'), indent=2)

    def _get_response_with_answer(self, author, ints, intents_json, message, answer):
        try:
            tag = ints[0]['intent']
            if tag == "i_dont_understand":
                self.learn_new_tag(intents_json, message, message, answer)
                return
            list_of_intents = intents_json
            for i in list_of_intents:
                if i['tag'] == tag:
                    if self.mood.get(author) is None:
                        self.mood[author] = Mood(self.model_name, author)
                    mood_change = self._get_mood_change(message)
                    self.mood.get(author).update_mood(mood_change[0], mood_change[1])
                    mood = self.mood.get(author).get_mood()
                    choice = mood + "_responses"
                    result = random.choice(i[choice])
                    break
            self.learn_new_input_and_answer(message, tag, answer)
            self.log_to_conversation(f"{result}, from tag: {tag}")
        except IndexError:
            result = "I don't understand!"
        self.history.put(('bot', result))
        print("learned_new_input: ", message, "for tag", )

    def learn_new_input(self, message, tag):
        with open(f'./sources/{self.model_name}/intents.json', 'r+') as f:
            data = json.load(f)
            for j in data:
                if j['tag'] == tag:
                    dictionary = j['patterns']
                    dictionary.append(message)
                    dictionary = sorted(list(set(dictionary)))
                    j['patterns'] = dictionary
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    f.truncate()
                    break

    def learn_new_input_and_answer(self, message, tag, answer):
        with open(f'./sources/{self.model_name}/intents.json', 'r+') as f:
            fields_to_edit = ['neutral_responses', "sad_responses", "happy_responses", 'depressed_responses',
                       'angry_responses', 'love_responses', 'best_friends_responses']
            data = json.load(f)
            for j in data:
                if j['tag'] == tag:

                    dictionary = j['patterns']
                    dictionary.append(message)
                    dictionary = sorted(list(set(dictionary)))
                    j['patterns'] = dictionary

                    for field in fields_to_edit:
                        dictionary = j[field]
                        dictionary.append(answer)
                        dictionary = sorted(list(set(dictionary)))
                        j[field] = dictionary
                    f.seek(0)
                    json.dump(data, f, indent=2)
                    break

    def _learn_from_new_type(self, intents_json, message):
        openai.api_key = OPEN_AI_API_TOKEN
        new_prompt = 'I am a highly intelligent question answering bot. If you ask me a question that is ' \
                     'rooted in truth, I will give you the answer. If you ask me a question that is nonsense,' \
                     ' trickery, or has no clear answer, I will respond with "Unknown".\n \n'
        history = self.history.get_all()
        history.reverse()
        for i, prompt in enumerate(history):
            if i >= 15:
                break
            elif isinstance(prompt, int):
                break
            elif prompt[0] == "bot":
                new_prompt += "A:" + prompt[1] + "\n"
            else:
                new_prompt += "Q:" + prompt[1] + "\n"

        new_prompt += "\nA:"

        start_time = time.perf_counter()

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=new_prompt,
            temperature=0,
            max_tokens=100,
            top_p=1,
            frequency_penalty=1.0,
            presence_penalty=2.0,
            stop=["\n"]
        )

        print('[DEBUG]: Violet Bot requested help from openAI at', datetime.now(), 'With message: "', new_prompt
              , '". The response took ', math.ceil((time.perf_counter() - start_time) * 1000),
              "milliseconds to calculate")

        result = response['choices'][0]['text']

        new_entry = {
            "tag": message,
            "patterns": [
                message
            ],
            "neutral_responses": [result],
            "sad_responses": [result],
            "happy_responses": [result],
            "depressed_responses": [result],
            "angry_responses": [result],
            "love_responses": [result],
            "best_friends_responses": [result],
        }

        intents_json.append(new_entry)
        self.history.put(("bot", result))
        json.dump(intents_json, open(f'./sources/{self.model_name}/intents.json', 'r+'), indent=2)
        return result

    def request_tag(self, message):
        pass

    def get_tag_by_id(self, id):
        pass

    def request_method(self, message):
        pass

    def request(self, author, message):
        self._store_message(author, message)
        ints = self._predict_class(message)

        if len(ints) == 0:
            return self._learn_from_new_type(self.intents, message)
        elif ints[0]['intent'] in self.intent_methods.keys():
            self.intent_methods[ints[0]['intent']]()
            return "wtf is this"
        else:
            return self._get_response(author, ints, self.intents, self.original_message)

    def train_with_prompts(self, message, correct_answer):
        self.log_to_conversation(f"\n {message}")
        self._store_message("training", message)
        ints = self._predict_class(message)

        if len(ints) == 0:
            self.learn_new_tag(self.intents, message, message, correct_answer)
        else:
            self._get_response_with_answer("training", ints, self.intents, self.original_message, correct_answer)

    def _store_message(self, username, message):
        self.history.put((username, message))

    def log_to_conversation(self, message):
        with open(f"sources/{self.model_name}/log_conversation.txt", "a") as f:
            f.write(f"{message} \n")


    def _get_mood_change(self, message):
        polarity_scores = self.sia.polarity_scores(message)
        result = [0, 0]
        for key, val in polarity_scores.items():
            if key == "pos":
                result[0] += val
            if key == "neg":
                result[0] -= val
            if key == "compound":
                result[1] += val
            if key == "neu":
                result[0] += val / 10

        return result
