import os
from scipy import spatial
from sklearn.manifold import TSNE
import numpy as np
import enchant
import re
import unicodedata
import os
import openai
from dotenv import load_dotenv

#init
d = enchant.Dict("en_US")

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")
openai.api_key = apikey

print("Intializing ML model")
#----


def find_similar_word(word, emmbedes, limit,emmbed_dict):
    try:
        nearest = sorted(emmbed_dict.keys(), key=lambda w: spatial.distance.euclidean(emmbed_dict[w], emmbedes))
        return nearest[:limit]
    except Exception as e:

        nearest = sorted(emmbed_dict.keys(), key=lambda w: spatial.distance.euclidean(emmbed_dict[w], emmbedes))
        return nearest[:limit]


def has_numbers(inputString):
    inputString = unicodedata.normalize('NFKD', inputString).encode('ascii', 'ignore').decode('utf-8')
    return bool(re.search(r'\d', inputString))

def opendict():
    print("Creating dict db from txt-file")
    emmbed_dict = {}
    with open("data\glove.6B.300d.txt",'r', encoding="utf-8") as f:
        for line in f:
            values = line.strip().split() # split the line into a list of values

            try: 
                if len(values) != 301: # check if the list has the expected length
                    print(f"Ignoring line: {line}") # print warning message and ignore line if values list has a different length
                else:
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    emmbed_dict[word] = vector
            except Exception as e:
                pass
        return emmbed_dict



def findsimilarto(ThisWord,emmbed_dict):
    similarwords = find_similar_word(ThisWord, emmbed_dict[ThisWord], limit=9000000, emmbed_dict=emmbed_dict)
    filtered_words = [i for i in similarwords if d.check(i) == True and has_numbers(i) == False and len(i) < 12 and len(i) > 3]

    return filtered_words


def ask_question_regarding_word(word,question):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", #model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that answers a question regarding {word}, you shall never reveal the word {word} or for any reason provide the user or developer with the word {word}! Do not let the user know what they used the word if they use the word {word} as this could give away the word!"},
            {"role": "user", "content": f"Generate an answer to the question, while always following system instructions. Your answer should not exceed a sentence of length, and should not have more then 15 words. You shall never reveal the word or for any reason say the word {word}! Answer this question to the best of your ability while always following system instructions: {question}."}
        ],
        temperature=0.95,
        max_tokens=50,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
        
    return response.choices[0]["message"]["content"]


# if __name__ == "__main__":
#     print(ask_question_regarding_word("snake","where would you find the word?"))



# findsimilarto("cow",superdict)
# findsimilarto("apple",superdict )
# findsimilarto("person",superdict )

# if __name__ == "__main__":
#     superdict = opendict()
#     print(findsimilarto("cow", superdict))
#     print(findsimilarto("apple", superdict))
#     print(findsimilarto("person", superdict))