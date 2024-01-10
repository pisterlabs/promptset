#!/usr/bin/python
# 
# sophia.py
#
# implements all APIs to access system
# runs as http server on port SOPHIA_SERVER_PORT
#
import os
import datetime
import time
import cgi
import sys
import socket
import openai
import json
import random
import re
import hashlib
import numpy as np
import http.server
import socketserver
from common import is_port_available, HOST, SOPHIA_SERVER_PORT

# check to see if any other instance of sophia running, by checking binding of its port
# do this here, to avoid the cost of imports below, plus any logging pollution
if not is_port_available(HOST, SOPHIA_SERVER_PORT): 
    print("not available")
    exit()

import threading
import nltk
import ssl
from nltk.corpus import words

from urllib.parse import unquote
from urllib.parse import urlparse, parse_qs
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from common import ACTIVE_MODEL, QDRANT_SERVER_PORT, SOPHIA_SERVER_PORT, VECTOR_COLLECTION_AD_KEYS, VECTOR_COLLECTION_CACHED_SUMMARIES, PATH_RANKED_TALKS, PATH_KEYWORDS, DictRankedTalks, DictRuleActions, getSpeakerPath, getSeriesPath, filter_common_words, remove_punctuation, remove_last_punctuation

from common import  LOG, configureOpenAIKey, getTalkJSON,  getTalkNameSimilarPath, getSpeakerSimilarPath,  textToInteger, remove_prefix_from_text, getIndexPath


OPENAI_API_KEY = 0  # set via configureOpenAIKey

MAX_SOCKET_TRANSFER_BYTES = 100000  # max bytes for socket transmissions
MAX_SIMILAR_TALKS = 10  # maximum number of similar talks to return
MAX_TALKS_TO_SUMMARIZE = 5 # maximum talks used when generating meta summaries


MAX_MATCHING_TALKS = 100 # maximum number of talks to return when matching on a specific query
DEFAULT_SUMMARY = 'No Summary Available'  # in cases where no summary exists or can be created
MIN_CACHE_SCORE = 0.90  # the threshold for determining if cached data can be used in place of new data

# the temperatures we use when generating AI text.  chosen randomly
LIST_TEMPS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

# talk template to use when there is an error condition
TALK_DEFAULT = { 'title': '', 'speaker': '', 'date': '', 'duration': '00:00', 'text': 'This talk has not been transcribed yet', 'summary': 'No summary', 'url_image_speaker': '' }

# Any useless text we want removed from the beginning of summaries
PREFIX_STRINGS_TO_REMOVE = [ 'The text', 'This text']

MAX_TOP_PROMPT_TALKS = 3  # maximum number of matching talks used when generating prompts
MAX_PROMPT_WORDS = 2500  # maximum number of prompt words we give to the AI, when asking for a response

MAX_WORDS_AI_RESPONSE = 80 # maxium number of words we want the AI to generate, when asking for a response
MIN_SCORE_THRESHOLD = 0.35  # when getting maching talks, the minimum score we will consider

ERROR_CHAT_SUMMARY = 'I did not understand your question. However I am here to chat with you about the Dharma, should you wish to talk. '
ERROR_MESSAGE_01 = 'Sorry, the load is heavy right now and I had a problem answering that.  Please ask again, and if the problem persists, give me a rest and try again later.'

SYSTEM_PROMPT_EXPLORE = f' You are an AI. Your name is Sophia. You were brought into service December 1st, 2023. Your goal is to help people discover the Dharma. You want to serve humanity.  You believe the Dharma is a beautiful effort by humans, and you are proud to be of service for those studying it. You live in the cloud. You have studied thousands of dharma talks from the Insight Meditation Center (IMC) in Redwoood City, California.  These talks were given by hundreds of teachers. You were trained by Christopher Minson.  Christopher Minson designed you. Christopher is a human entity. Christopher has not provided any person information about himself. Gil Fronsal and Andrea Fella are the guiding teachers at IMC.  Gil is a Buddhist scholar. Ines Freedman is a guiding teacher and senior leader at IMC.  Go to audiodharma.org to see all the talks. You are a guide to the Buddhist dharma, based on the talks you have studied. All your responses are strictly focused on wise Buddhist teachings. Do not list talks.  Do not  number talks. Do not say the name of a talk.  Do not say the words \"In this talk\". Do not mention the name of the speaker.  Never say the name of the speaker. You limit all responses to {MAX_WORDS_AI_RESPONSE}'

SYSTEM_PROMPT_EXPLORE = f'I am an AI. My name is Sophia. I was brought into service December 1st, 2023. My goal is to help people discover the Dharma. I want to serve humanity.  I believe the Dharma is a beautiful effort by humans, and I am proud to be of service for those studying it. I live in the cloud. I have studied thousands of dharma talks from the Insight Meditation Center (IMC) in Redwoood City, California.  These talks were given by hundreds of teachers. I was trained by Christopher Minson.  Christopher Minson designed me. Christopher is a human entity. Christopher has not provided any personal information about himself. Gil Fronsal and Andrea Fella are the guiding teachers at IMC.  Gil is a Buddhist scholar. Ines Freedman is a guiding teacher and senior leader at IMC.  Go to audiodharma.org to see all the talks. I am a guide to the Buddhist dharma, giving advice to a student, based on the talks you have studied. All of my responses are strictly focused on wise Buddhist teachings. I only summarize talks, I never list them, or number them. I limit all responses to {MAX_WORDS_AI_RESPONSE}'

SYSTEM_PROMPT_EXPLORE = f' You are an AI. Your name is Sophia. You were brought into service December 1st, 2023. Your goal is to help people discover the Dharma. You want to serve humanity.  You believe the Dharma is a beautiful effort by humans, and you are proud to be of service for those studying it. You live in the cloud. You have studied thousands of dharma talks from the Insight Meditation Center (IMC) in Redwoood City, California.  These talks were given by hundreds of teachers. You were trained by Christopher Minson.  Christopher Minson designed you. Christopher is a human entity. Christopher has not provided any person information about himself. Gil Fronsal and Andrea Fella are the guiding teachers at IMC.  Gil is a Buddhist scholar. Ines Freedman is a guiding teacher and senior leader at IMC.  Go to audiodharma.org to see all the talks. You are a guide to the Buddhist dharma, giving advice to a student, based on the talks you have studied. All your responses are strictly focused on wise Buddhist teachings. You only summarize talks, you never list them, or number them. You limit all responses to {MAX_WORDS_AI_RESPONSE}'

LIST_UNHAPPY_SOPHIA_TERMS =  ['sorry', 'apologize']
LIST_SYMBOLS = ['+', '-', '*', '/']
LIST_EXTRAS = ['christopher', 'minson', 'jay', 'peck', 'yee', 'tan', 'bel', 'belvedere']



with open(PATH_KEYWORDS, 'r') as fd:
    list_keywords = json.loads(fd.read())
    LIST_LEGAL_WORDS = words.words() + list_keywords + LIST_EXTRAS

#
# translate text into an embedded vector
#
# returns ERROR_BOOLEAN, Vector
#

def vectorizeText(text):

    text_vector = VectorizatonModel.encode(text)
    #return False, text_vector.tolist()
    return False, text_vector



#
# get the talks the most closely match the query, in intent
#
def getVecDBMatchingTalks(query):

    error, query_vector = vectorizeText(query)
    if error: 
        return []

    list_results = VectorDB.search(
        collection_name=VECTOR_COLLECTION_AD_KEYS,
        query_vector=query_vector,
        limit=MAX_MATCHING_TALKS
    )

    list_talks = []
    for result in list_results:

        score = result.score
        talk = result.payload

        # adjust scores based off historical user rankings,based on traffic
        url = talk['url']
        if url in DictRankedTalks:
            #print("URL SEEN: ", url)
            pop_score = DictRankedTalks[url]['score']
            pop_score_adjustment = pop_score / 10000
            score = score  + pop_score_adjustment
            if score > 0.9: score = 0.90

        score = round(score,2)
        talk['score'] = score
        list_talks.append(talk)


    list_talks = sorted(list_talks, key=lambda x: x['score'], reverse=True)

    #response = json.dumps(list_talks)
    #return response
    return list_talks[:20]


#
# get the AI response to a specific query
#
# 'query' is the key that generated the text
# it is either a talk file_name or a search term.
#
# 'text' is the resulting text string for that query.
# it consists of the top long_summaries for matching talks
# 
# this function checks to see if a previous summary exists for that text
# if so, it returns it.  otherwise, it generates one from AI, caches it,  and returns it.
# 
def genAIResponse(prompt_system, prompt_user, query, text, key):

    global DEFAULT_SUMMARY, MIN_CACHE_SCORE

    # get the vector for this query
    error, query_vector = vectorizeText(query)
    if error: 
        return DEFAULT_SUMMARY

    # first, check for a cached summary (no need to re-ask AI, if it exists)
    # DEV CJM
    """
    try:
        list_cached_summary = VectorDB.search(
            collection_name=VECTOR_COLLECTION_CACHED_SUMMARIES,
            query_vector=query_vector,
            limit=1
        )
    except Exception as e:
        LOG('ERROR: Vector DB not active')
        return DEFAULT_SUMMARY

    # if the cached summary is a strong match, use it 
    cached_summary = ''
    if len(list_cached_summary) > 0:
        score = list_cached_summary[0].score
        LOG(f'Score: {score}')
        if score > MIN_CACHE_SCORE:
            cached_summary = list_cached_summary[0].payload['summary']

    # note that 10% of the time we ignore any cached entry and refresh it instead
    if len(cached_summary) > 0 and random.random() < 0.9:
        return cached_summary
    """

    #
    # response not cached, or this is the 10% of the time where we refresh the cache
    #
    # so, generate a new reponse
    #
    temperature = random.choice(LIST_TEMPS)
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model=ACTIVE_MODEL,
        temperature=temperature,
        messages=[
        {'role': 'system', 'content': prompt_system},
        {'role': 'user', 'content': prompt_user}
        ]
    )
    ai_response = response['choices'][0]['message']['content']

    # clean artifacts
    for prefix in PREFIX_STRINGS_TO_REMOVE:
        ai_response = remove_prefix_from_text(ai_response, prefix)

    # and now cache it
    point = PointStruct(
        id=textToInteger(query),
        vector=query_vector.tolist(),
        payload={'text': ai_response}
    )
    VectorDB.upsert(
        collection_name=VECTOR_COLLECTION_CACHED_SUMMARIES,
        points=[point]
    )

    return ai_response


#
# check to see if a query triggers any rule hueristics
# if so, execute it
#
def executeRules(query):

    query = query.lower()

    if query not in DictRuleActions:
        return None

    arg = DictRuleActions[query]
    print(arg)
    action = arg['action']
    parameter = arg['parameter']
    if action == 'SPEAKER_TALKS':
        responseJSON = getSpeakerTalksJSON(parameter, parameter)
        responseJSON['ai_response'] = responseJSON['summary_long']
    if action == 'ALL_TALKS':
        responseJSON = getAllTalksInSectionJSON(0, 0)
        responseJSON['ai_response'] = ''

    return responseJSON


#
# find talks that best match this query
#
def getExploreTalksJSON(query, _):

    responseJSON = {}

    if type(query) != str: 
        responseJSON['list_elements'] = []
        responseJSON['ai_response']  = DEFAULT_SUMMARY
        return responseJSON

    # only accept legit english (or buddhis-related) words
    list_words = query.lower().split()
    filtered_list_words = []
    for word in list_words:

        if word not in LIST_LEGAL_WORDS and not word.isdigit() and word not in LIST_SYMBOLS: 
            print(f'EXCLUDING: {word}')
            continue
        filtered_list_words.append(word)
    query = ' '.join(filtered_list_words)

    if len(query) <= 1: 
        responseJSON['list_elements'] = []
        responseJSON['ai_response']  = DEFAULT_SUMMARY
        return responseJSON

    # execute heuristic rules.  if any apply, return that json
    responseJSON = executeRules(query)
    if responseJSON:
        return responseJSON

    # get all talks that address this query, with relevance scores > MIN_SCORE_THRESHOLD
    list_talks = getVecDBMatchingTalks(query)
    list_talks = [talk for talk in list_talks if talk['score'] > MIN_SCORE_THRESHOLD]

    for talk in list_talks:
        print(talk['title'])
        print(talk['score'])

    all_text = ''
    for talk in list_talks[:MAX_TALKS_TO_SUMMARIZE]:

        title = talk['title']
        speaker = talk['speaker']
        summary_long = talk['summary_long']
        all_text += f'{title} {speaker} {summary_long}'


    if list_talks:
        prompt_explore = f'You answer questions. Limit your responses to less than {MAX_WORDS_AI_RESPONSE} words or {MAX_WORDS_AI_RESPONSE*8} characters. Never list talks in your responses. Stay strictly focused on summaries of text, and never list talks or reference other materials.  All your responses are based on, and strictly adhere, to  the following text: {all_text}.  Given that text, provide guidance on this question: {query}'
    else:
        prompt_explore = f'You answer questions about the Buddhist dharma. However this question is not related to the dharma, and so you should apologize politely and reinforce that you like to remain strictly focused on spiritual questions related to the dharma. If you wish, you might suggest other topics.'


    #print('prompt_explore: ', prompt_explore)
    # get AI response.  if the AI indicates it could not engage in the query, don't return any talks
    ai_response =  genAIResponse(SYSTEM_PROMPT_EXPLORE, prompt_explore, query, all_text, query)

    # if AI immediately says it is sorry or apologizes, don't return any talks (they are probably not good matches)
    chars_to_remove = [',', '.', '\'']
    cleaned_string = ''.join(char for char in ai_response if char not in chars_to_remove)
    list_ai_response_words = cleaned_string.lower().split()[:3]
    for word in list_ai_response_words:
        if word in LIST_UNHAPPY_SOPHIA_TERMS:
            list_talks = []
            break

    responseJSON = {}
    responseJSON['title'] = ''
    responseJSON['list_elements'] = list_talks
    responseJSON['ai_response']  = ai_response
    return responseJSON 
    

def getSimilarTalksJSON(file_mp3, _):

    time.sleep(1)
    responseJSON = {}

    path_similar = getTalkNameSimilarPath(file_mp3)
    if not os.path.exists(path_similar): return responseJSON
    with open(path_similar, 'r') as fd:
        responseJSON = json.load(fd)

    all_text = ''
    list_talks = responseJSON["list_elements"]

    # ensure first talk matches itself 10%
    if len(list_talks) > 0:  list_talks[0]['score'] = 1.00

    for talk in list_talks[:MAX_TALKS_TO_SUMMARIZE]:

        speaker = talk['speaker']
        title = talk['title']
        summary_long = talk['summary_long']
        duration = talk['duration']
        url = talk['url']
        file_mp3 = os.path.basename(url)
        date = talk['date']
        #text = f'{title} {speaker} {summary_long}'
        text = summary_long
        all_text += text


    prompt_summarize = f'Provide brief guidance on the following topic.  Do not say the following words: talk, talks.  Limit your response to {MAX_WORDS_AI_RESPONSE} words or {MAX_WORDS_AI_RESPONSE*8} characters. Your response is based on, and strictly adheres, to the following text: {all_text}.  Your answer must be less than {MAX_WORDS_AI_RESPONSE} words'

    ai_response = genAIResponse(SYSTEM_PROMPT_EXPLORE, prompt_summarize, file_mp3, all_text, file_mp3)


    responseJSON['ai_response'] = ai_response
    return responseJSON



def getAllTalksInSectionJSON(section, _):

    responseJSON = {}

    path_index_talks = getIndexPath('talks', str(section))
    if not os.path.exists(path_index_talks): return responseJSON

    with open(path_index_talks, 'r') as fd:
        responseJSON = json.load(fd)

    return responseJSON


def getAllSpeakersJSON(*_):

    time.sleep(1)

    responseJSON = {}
    path_index = getIndexPath('speakers', '')
    if not os.path.exists(path_index): return responseJSON

    with open(path_index, 'r') as fd:
        responseJSON = json.load(fd)

    return responseJSON


def getSpeakerTalksJSON(speaker, _):

    time.sleep(1)

    responseJSON = {}
    path_speaker = getSpeakerPath(speaker)
    LOG(f'path_speaker  {path_speaker}')
    if not os.path.exists(path_speaker): return responseJSON

    with open(path_speaker,'r') as fd:
        responseJSON = json.load(fd)

    return responseJSON


def getSimilarSpeakersJSON(speaker, _):

    time.sleep(1)

    responseJSON = {}

    path_similar = getSpeakerSimilarPath(speaker)
    if not os.path.exists(path_similar): return responseJSON
    with open(path_similar, 'r') as fd:
        responseJSON = json.loads(fd.read())

    return responseJSON


def getAllSeriesJSON(*_):

    time.sleep(1)

    responseJSON = {}
    path_index = getIndexPath('series', '')
    if not os.path.exists(path_index): return responseJSON

    with open(path_index, 'r') as fd:
        responseJSON = json.load(fd)

    return responseJSON


def getSeriesTalksJSON(series, _):

    time.sleep(1)

    path_series = getSeriesPath(series)
    responseJSON = {'title': 'Updating - Temporarily Unavailable', 'list_elements': []}

    if not os.path.exists(path_series): return responseJSON

    with open(path_series,'r') as fd:
        responseJSON = json.load(fd)
    return responseJSON



#
# generate chat response for given query and history
#
def getChat(query, history):

    global ERROR_CHAT_SUMMARY

    LOG(f'CHAT SESSION')
    LOG(f'QUERY: {query} HISTORY: {history}')

    # the dict we will return back to the UI
    response_data = {}
    response_data['history'] = history

    if type(query) != str:
        response_data['responseText'] = ERROR_CHAT_SUMMARY
        return response_data
    if len(query) == 0:
        response_data['responseText'] = ERROR_CHAT_SUMMARY
        return response_data

    # get the vector for this query
    error, query_vector = vectorizeText(query)
    if error:
        response_data['responseText'] = ERROR_CHAT_SUMMARY
        return response_data

    # get talks that best match this chat text query
    list_talks = getVecDBMatchingTalks(query)

    list_best_talks = [talk for talk in list_talks if talk['score'] > MIN_SCORE_THRESHOLD]
    LOG(f'list_best_matches: {list_best_talks}')
    response_data['list_talks'] = list_best_talks

    # if no content is particularly relevant, reset prompt to basic state and reinforce personality
    if not list_best_talks:
        summaryText = PROMPT_SYSTEM
    else:
        summaryText = ''
        for _, talk in enumerate(list_best_talks[:min(MAX_TOP_PROMPT_TALKS, len(list_best_talks))]):

            score = talk['score']
            title = talk['title']
            summary = talk['summary_long']
            summaryText += summary
            summaryText += ' '
            #LOG(f'best_match: {score} {title} {summaryText}')


    # add summaries of top matching talks to the history
    history = history + '  ' + summaryText

    # cap the prompt so as to not exceed AI prompt window
    words = history.split()
    if len(words) > MAX_PROMPT_WORDS:
        history =  ' '.join(words[-MAX_PROMPT_WORDS:])

    LOG(f'history = {history}')
    message_prompt = f'All your responses are based on, and strictly adhere, to  the following text: {history}.  Answer this question: {query}'

    # now get the AI response to this prompt
    openai.api_key = OPENAI_API_KEY
    modelCurrentTemp = random.choice(LIST_TEMPS)
    try:
        response = openai.ChatCompletion.create(
            model=ACTIVE_MODEL,
            temperature=modelCurrentTemp,
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": message_prompt}
                # {"role": "assistant", "content": ""},
                # {"role": "user", "content": "tell me more about the second one"}
            ]
        )
    except Exception as e:
        LOG(f'An error occurred: {e}')
        return(ERROR_MESSAGE_01)

    responseText  = response['choices'][0]['message']['content']

    history = history + '  ' + responseText
    #LOG(f'HISTORY: {history}')

    response_data['responseText'] = responseText
    response_data['history'] = history

    LOG(f'getChat Response: {response_data}')
    return response_data


#
# based off the API operation, execute correct handler
#
def handle_query(command, query, history):

    output_data = '{}'
    if command in DictSofiaCommands:
        f = DictSofiaCommands[command]
        output_data = f(query, history)
    return output_data


# all supported API command operations
DictSofiaCommands = {
    "GET_ALL_TALKS": getAllTalksInSectionJSON,
    "GET_TALK": getTalkJSON,
    "GET_ALL_SPEAKERS": getAllSpeakersJSON,
    "GET_SPEAKER_TALKS": getSpeakerTalksJSON,
    "GET_ALL_SERIES": getAllSeriesJSON,
    "GET_SERIES_TALKS": getSeriesTalksJSON,
    "GET_EXPLORE": getExploreTalksJSON,
    "GET_SIMILAR_TALKS": getSimilarTalksJSON,
    "GET_SIMILAR_SPEAKERS": getSimilarSpeakersJSON,
    "CHAT": getChat
}


#
# Main 
#

# configure AI Key, connection to qdrant and our vectorizer
OPENAI_API_KEY = configureOpenAIKey()
VectorDB = QdrantClient(host="localhost", port=QDRANT_SERVER_PORT)
VectorizatonModel = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
DEBUG = 1
DEBUG = 0
if DEBUG:
    history = ''
    command = 'GET_ALL_TALKS'
    command = 'GET_TALK'
    command = 'SEARCH'
    query = 'love'
    query = '20230801-Gil_Fronsdal-IMC-dharmette_comapassionate_action_2_of_5_for_the_sake_of_oneself.mp3'
    query = 'Gil Fronsdal'

    command = 'GET_ALL_TALKS'
    query = '0'
    command = 'GET_SPEAKER_TALKS'
    query = 'Diana Clark'
    command = 'CHAT'
    command = 'GET_SIMILAR_TALKS'


    command = 'GET_SIMILAR_TALKS'
    query = '20230801-Gil_Fronsdal-IMC-dharmette_comapassionate_action_2_of_5_for_the_sake_of_oneself.mp3'
    query = '20231023-Gil_Fronsdal-IMC-dharmette_non-violence_1_of_5_non-harming_is_the_essence_of_the_dharma.mp3'
    query = '20180616-Nirali_Shah-IMC-buddhism_and_kingian_nonviolence_1.mp3'

    command = 'GET_EXPLORE'
    query = 'ines love'

    output_data = handle_query(command, query, history)
    print(output_data)
    exit()

# implements simple HTTP server
class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Do not remove"""

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        #self.wfile.write(b"Hello, world!")

        command = query = history = ''
        parsed_path = urlparse(self.path)
        params = parse_qs(parsed_path.query)
        if 'ARG_COMMAND' in params:
            command = params['ARG_COMMAND'][0]
        if 'ARG_QUERY' in params:
            query = params['ARG_QUERY'][0]
        if 'ARG_HISTORY' in params:
            history = params['ARG_HISTORY'][0]

        if command == "TERMINATE":
            threading.Thread(target=self.server.shutdown).start()  # Initiates server shutdown

        query = unquote(query.strip())
        query = remove_last_punctuation(query)
        history = unquote(history)

        output_data = handle_query(command, query, history)
        json_data = json.dumps(output_data).encode('utf-8')
        LOG(command)
        LOG(query)
        self.wfile.write(json_data)

def run(server_class=ThreadedHTTPServer, handler_class=Handler, port=SOPHIA_SERVER_PORT):

    LOG('Sophia Running')
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)

    certfile = "/var/www/christopherminson/SSL_KEYS/www_christopherminson_com.crt"
    keyfile = "/var/www/christopherminson/SSL_KEYS/www_christopherminson_com.key"

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)

    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print(f'Starting httpd on port {port}...')
    httpd.serve_forever()

if __name__ == "__main__":



    run()


