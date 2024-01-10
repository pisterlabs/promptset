import os
from langchain.llms import OpenAI
import yake
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get('openai_api_key')

# To generate a script for the video.
def get_storyline(question):
    llm = OpenAI(max_tokens=1024)
    response = llm.predict(question)
    print(response)
    return response

# question = "What would happen if the sun starts to disintegrate in our solar system? Could you describe the series of events as well as the physics behind it? Not more than 3 sentences."
# get_storyline(question)

# Extract the sentences in the script. Each sentence will use 1 video.
def get_sentences(storyline):
    sentences = storyline.split(".")
    sentences_lst = []
    for i in sentences:
        sentences_lst.append(i.strip())
    
    print(sentences_lst[:-1])
    return sentences_lst[:-1]

# Extract keywords from the sentence to make it shorter.
# FIXME: not working too well. can use OpenAI to extract the main objects
def get_reduced_sentence(sentence):
    kw_extractor = yake.KeywordExtractor(
        lan = "en", 
        n = 2,            # max_ngram_size
        dedupLim = 0.9,   # deduplication_threshold
        top = 3,          # numOfKeywords
        features = None
        )

    keywords = kw_extractor.extract_keywords(sentence)
    
    reduced_sentence = ""
    for kw in keywords:
        print(kw[0])
        reduced_sentence += kw[0] + " "

    print(reduced_sentence)
    return reduced_sentence

# sentence = """A conspiracy theory that is fictional is that crop circles are created by aliens as a way to communicate and connect with humans"""
# keywords(sentence)