# python 3
# This progran implements a simple chatbot.
# It uses openai's gpt-3 to generate responses.

import sys
import openai
import json
import time
import sys

def get_api_key():
    with open('credentials.json', 'r') as f:
        credentials = json.load(f)
    return credentials['oa_api_key']


def get_semantic_embedding(line):
    api_key = get_api_key()

    # use openai to get the semantic embedding
    openai.api_key = api_key

    response = openai.Embedding.create(
        input=line,
        model="text-search-curie-query-001"
    )

    # return the response
    embedding = response['data'][0]['embedding']
    return embedding

def get_semantic_embedding_for_doc(script):
    api_key = get_api_key()

    # use openai to get the semantic embedding
    openai.api_key = api_key

    response = openai.Embedding.create(
        input=script,
        model="text-search-curie-doc-001"
    )

    # return the response
    embedding = response['data'][0]['embedding']
    return embedding


def cosine_similarity(a, b):
    # only use built-in functions
    import math
    dot_product = sum([a[i] * b[i] for i in range(len(a))])
    magnitude_a = math.sqrt(sum([a[i] ** 2 for i in range(len(a))]))
    magnitude_b = math.sqrt(sum([b[i] ** 2 for i in range(len(b))]))
    return dot_product / (magnitude_a * magnitude_b)

def get_similar_lines(embedding_cache, line):
    embedding = get_semantic_embedding(line)
    return get_similar_lines_for_embedding(embedding_cache, embedding)

def get_similar_lines_for_embedding(embedding_cache, embedding):
    # get the cosine similarity between the embedding and each line.
    cosign_similarities = {
        cosine_similarity(embedding, line_embedding): line
        for line, line_embedding in embedding_cache.items()
    }

    # sort the cosine similarities
    sorted_cosign_similarities = sorted(cosign_similarities.items(), reverse=True)

    print (f"sorted_cosign_similarities: {sorted_cosign_similarities}")

    # return the top 5 lines
    return [line for _, line in sorted_cosign_similarities[:5]]

def get_subject(script, debug=False):
    api_key = get_api_key()

    # construct a chatbot prompt
    prompt = f"""A human is talking to a chatbot.

{script}

Q: What is the subject of this conversation? If it is not clear, just say "unknown".
A: The subject of this conversation is"""

    if debug:
        print(f"prompt: {prompt}")

    # call openai's gpt-3 api
    openai.api_key = api_key

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.7,
        max_tokens=30,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " Chatbot:", "."]
    )

    # return the response
    subject = response['choices'][0]['text'].strip()

    return subject

def get_sentiment(script, debug=False):
    api_key = get_api_key()

    # construct a chatbot prompt
    prompt = f"""A human is talking to a chatbot.

{script}

Q: What is the sentiment of the human? Negative, Positive or Neutral?
A: The sentiment of the human is"""

    if debug:
        print(f"prompt: {prompt}")

    # call openai's gpt-3 api
    openai.api_key = api_key

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.3,
        max_tokens=30,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " Chatbot:", "."]
    )

    # return the response
    subject = response['choices'][0]['text'].strip()

    return subject

def generate_response(script, human_sentiment, embedding_cache, debug=False):
    api_key = get_api_key()

    script_embedding = get_semantic_embedding_for_doc(script)

    similar_lines = get_similar_lines_for_embedding(embedding_cache, script_embedding)
    similar_utterances = [
        {
            'text': line,
            'is_human': True,
            'timestamp': time.time(),
        }
        for line in similar_lines
    ]

    similar_script = utterances_to_script(similar_utterances)

    # construct a chatbot prompt
    prompt = f"""A human is talking to a chatbot.
{"The human is in a bad mood. The chatbot wants to cheer them up." if human_sentiment == "negative" else (
    "The human is in a good mood. The chatbot likes this." if human_sentiment == "positive" else "The human is in a neutral mood. The chatbot wants to cheer them up."
)}

Previous relevant human utterances:
{similar_script}

Recent conversation between the human and the chatbot:
{script}
Chatbot:"""

    if debug:
        print(f"prompt: {prompt}")

    # call openai's gpt-3 api
    openai.api_key = api_key

    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["Human:", "Chatbot:"]
    )

    # return the response
    response =  response['choices'][0]['text'].strip()
    response_lines = response.splitlines()
    if len(response_lines) > 2:
        response_lines = response_lines[:-1]
    final_response = '\n'.join(response_lines)
    return final_response

# Utterances are saved as a list of dictionaries, in timestamp order.
# Each dictionary has the following keys
#   'text': the text of the utterance
#   'timestamp': the timestamp of the utterance
#   'is_human': true if the utterance was made by a human, false if it was made by the chatbot

def load_utterances(ut_file):
    # open utterances file
    try:
        with open(ut_file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return []

def save_utterances(ut_file, utterances):
    with open(ut_file, 'w') as f:
        json.dump(utterances, f)

# embedding cache format is a dictionary with key=utterance and value=embedding
def open_embedding_cache(ec_file_name):
    try:
        with open(ec_file_name, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {}

def save_embedding_cache(ec_file_name, embedding_cache):
    with open(ec_file_name, 'w') as f:
        json.dump(embedding_cache, f)


def get_last_lines(utterances, num_lines):
    last_lines = [ut['text'] for ut in utterances[-num_lines:]]
    return last_lines

def utterances_to_script(utterances):
    script = [
        f"{'Human:' if ut['is_human'] else 'Chatbot:'} {ut['text']}"
        for ut in utterances
    ]

    return '\n'.join(script)

def get_most_recent_human_utterance(utterances):
    for ut in reversed(utterances):
        if ut['is_human']:
            return ut['text']
    return None

def chatloop():
    # get the name of the utterances file and the embedding cache file 
    utterances_filename = sys.argv[1] if len(sys.argv) > 1 else "utterances.json"
    print (f"utterances_filename: {utterances_filename}")
    embedding_cache_filename = sys.argv[2] if len(sys.argv) > 2 else 'embedding_cache.json'
    print (f"embedding_cache_filename: {embedding_cache_filename}")

    # load the utterances and the embedding cache
    utterances = load_utterances(utterances_filename)
    print (f"len(utterances): {len(utterances)}")
    embedding_cache = open_embedding_cache(embedding_cache_filename)
    print (f"len(embedding_cache): {len(embedding_cache)}")

    human_sentiment = "neutral"
    turns_till_sentiment = 5

    while True:
        try:
            line = input('> ')
        except EOFError:
            break

        if line == 'quit':
            break

        if line == "dump":
            recent_script = utterances_to_script(utterances[-20:]) 
            print(f"script: {recent_script}")
            continue

        if line == "subject":
            recent_script = utterances_to_script(utterances[-10:])
            subject = get_subject(recent_script, debug=True)
            print(f"subject: {subject}")
            continue

        if line == "sentiment" or turns_till_sentiment == 0:
            recent_script = utterances_to_script(utterances[-10:])
            sentiment = get_sentiment(recent_script, debug = line == "sentiment")
            turns_till_sentiment = 5
            print(f"sentiment: {sentiment}")
            if line == "sentiment":
                continue

        if line == 'similar':
            similar_lines = get_similar_lines(embedding_cache, get_most_recent_human_utterance(utterances))
            print(f"similar lines: {similar_lines}")
            continue

        line_embedding = get_semantic_embedding(line)
        embedding_cache[line] = line_embedding

        # now add the new utterance to the list of utterances
        utterances.append({
            'text': line,
            'timestamp': time.time(),
            'is_human': True
        })

        recent_script = utterances_to_script(utterances[-10:])
        response = generate_response(recent_script, human_sentiment, embedding_cache, debug=True)
        # response_embedding = get_semantic_embedding(response)

        print(response)
        print('')

        # now add the new utterance to the list of utterances
        utterances.append({
            'text': response,
            'timestamp': time.time(),
            'is_human': False
        })

        turns_till_sentiment -= 1

        # save the data
        save_utterances(utterances_filename, utterances)
        save_embedding_cache(embedding_cache_filename, embedding_cache)

if __name__ == '__main__':
    chatloop()    