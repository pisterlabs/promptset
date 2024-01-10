"""
Mohamed Gufran, 000836816

This is a chat bot program which takes the help of file_input to read the question and answer from the .txt file uses vector representation as well as using a classifier to generate an appropriate response apart from that
it also use a different classifier and vectorizer which is imported from outside to determine the emotional tone
of the user as well as it use gpt3 transformer api to generate unique responses when the intent was not matched with any of the question

resources I used for the answer -- sentiment.py
"""
## importing the necessary files
import openai
from file_input import *
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from api_key import api_key

## loading the trained classifer and vectorizer of another program
clf1 = load("toneClassifier.joblib")
vectorizer2 = load("toneVectorizer.joblib")

openai.api_key = api_key #api key for the gpt3 transformer


vectorizer = CountVectorizer(stop_words='english', max_df=0.9) #vectorizer i used for the intent matching

## loading the question and answer
def load_FAQ_data():
    """Thi s method returns a list of questions and answers. The
    lists has two questions for each answer."""

    question = file_input("question.txt")

    answers = file_input("answers.txt")



    return question, answers

## either determining the intent or generating the emotional tone
def understand(utterance):
    """This method processes an utterance to determine which intent it
    matches. The index of the intent is returned, or -1 if the intent doesn't match
    which than generates the unique response from the gpt3 transformer,
    and even check for the emotional tone when the utterance have zero cosine similarity
    with the questions."""

    global intents # declare that we will use a global variable

    try:
        vectors = vectorizer.fit_transform(intents) #for intent matching
        new_vector = vectorizer.transform([utterance])

        vc1 = vectorizer2.transform([utterance]) # for checking emotional tone
        tone = clf1.predict(vc1)


        cosine_sim = cosine_similarity(new_vector, vectors)

        similarities = cosine_sim[0]
        best = 0
        t = 0.57

        #for intent matching
        for i in range(len(similarities)):
            if similarities[i] > similarities[best]:
                if similarities[i] >= t:
                    best = i
        # print(similarities)
        # print(similarities[best])
        # print(best)
        # print(tone)

        # checking if utterence is just a little bit similar to topic if yes then return -1
        for i in range(len(similarities)):
            if similarities[i] >= 0.35 and  similarities[i] < t and similarities[best] == 0.0:
                print("hh")
                return -1

        # for checking emotional tone
        if  similarities[best] == 0:
            if tone[0] == 0:
                best = -2       #-2 for negative utterance
            elif tone[0] == 1:
                best = -3       #-3 for positive utterance
        return best
    except ValueError:
        return -1

## Generating the responses
def generate(intent):
    """This function returns an appropriate response given a user's
    intent."""

    global responses # declare that we will use a global variable

    # user friendly response when user is upset or happy and types something out of the scope
    if intent == -2:
        return "I am really sorry to upset you I think I don't have any information about this \r\nif you want you can go to wikipedia - https://en.wikipedia.org/wiki/Ahmedabad \r\nor else you can go to the Ahmedabad tourism website - https://www.gujarattourism.com/central-zone/ahmedabad.html "
    elif intent == -3:
        return "thank you very much I will take it as an appreciation"
    return responses[intent]


## Load the questions and responses
intents, responses = load_FAQ_data()

gpt3_file = open("gpt3.txt")
gpt3_answers = gpt3_file.read()
gpt3_file.close()


## Main Program

def chat():
    # talk to the user
    print("Hello! I know stuff about chat bots. When you're done talking, just say 'goodbye' or 'quit'.")
    print()
    utterance = ""
    while True:
        utterance = input(">>> ")
        if (utterance == "goodbye" or utterance == "quit"):
            break;
        intent = understand(utterance)
        # if the utterance is about the topic but doesn't match with the question then use gpt3 transformer
        if intent == -1:
            answer = openai.Completion.create(engine="davinci", prompt=gpt3_answers+"\nQ22) "+utterance+"\nA22)", max_tokens=200, temperature=0.5, stop=["\n"])

            response = "I am not 100% sure about this but the answer may be: \r\n" + answer["choices"][0]["text"]
            print(response)
        else:
            response = generate(intent)
            print(response)
        print()

    print("Nice talking to you!")
