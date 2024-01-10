import os
import openai
from time import time,sleep
import numpy as np


with open('openaiapikey.txt', 'r') as infile:
    open_ai_api_key = infile.read()
openai.api_key = open_ai_api_key


def save_memory(content, label):
    filename = 'nexus/%s_%s.txt' % (time(), label)
    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def get_timestamp(filename):
    return float(filename.split('_')[0])


tempo = 30
context_prompts = [('prompt_cof.txt', 'Core Objective Functions: ', 'COF'),
                   ('prompt_task.txt', "Raven's task: ", 'task'),
                   ('prompt_empathy.txt', "Raven's empathy: ", 'empathy'),
                   ('prompt_prediction.txt', "Raven's predictions: ", 'prediction'),
                   ('prompt_consequences.txt', "Consequences of Raven's actions: ", 'consequences'),
                   ('prompt_questions.txt', "Important questions: ", 'questions'),
                   ('prompt_teach.txt', "Important information: ", 'teach')]


# TODO - declarative memory?


control_prompts = [('prompt_envision.txt', "Envision my goal state: ", 'envision'),  # envision what the goal looks like
                   ('prompt_brainstorm.txt', "Potential goal steps: ", 'brainstorm'),  # brainstorm actions
                   ('prompt_execution.txt', "My next action: ", 'execution')]


metacognition_prompts = [('prompt_persona.txt', "Raven's personality: ", 'persona'),
                         ('prompt_metacognition.txt', "Raven's metacognition: ", 'metacognition'),
                         ('prompt_philosophy.txt', "Philosophical rumination: ", 'philosophy')]


def stack_memories(memories):
    results = list()
    now = time()
    for memory in memories:
        timestamp = float(memory['filename'].split('_')[0])
        age = now - timestamp
        if age < 60:  # less than a minute, so measure seconds
            line = '%s seconds ago: %s' % (int(age), memory['content'])
        elif age < 3600:  # less than an hour, so measure minutes
            line = '%s minutes ago: %s' % (int(age/60), memory['content'])
        elif age < 86400:  # less than day, so measure hours
            line = '%s hours ago: %s' % (int(age/3600), memory['content'])
        else:  # more than a day ago, so measure days
            line = '%s days ago: %s' % (int(age/86400), memory['content'])
        results.append(line)
    return '\n'.join(results)


def generate_prompt(prompt_file, logs):
    prompt = read_file(prompt_file).replace('<<LOGS>>', logs)
    return gpt3_completion(prompt, engine='text-davinci-002')


def gpt3_embedding(content, engine='text-similarity-ada-001'):
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    save_gpt3_log(content, str(vector))
    return vector


def similarity(v1, v2):  # return dot product of two vectors
    return np.dot(v1, v2)


def search_index(text, nexusindex, count=5, olderthan=None):
    vector = gpt3_embedding(text)
    scores = list()
    for i in nexusindex:
        if i['vector'] == vector:  # this is identical, skip it
            continue
        if olderthan:
            timestamp = get_timestamp(i['filename'])
            if timestamp > olderthan:
                continue
        score = similarity(vector, i['vector'])
        scores.append({'filename': i['filename'], 'score': score})
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    results = list()
    for i in ordered:
        results.append({'filename': i['filename'], 'score': i['score'], 'content': read_file('nexus/'+i['filename'])})
    if len(results) > count:
        return results[0:count]
    else
        return results


def latest_memories(count):
    files = os.listdir('nexus/')
    if len(files) > count:
        files = files[-count:]
    result = list()
    for file in files:
        content = read_file('nexus/' + file)
        result.append({'filename': file, 'content': content})
    return result


def file_exists(filename, data):
    for i in data:
        if filename == i['filename']:
            return True
    return False


def update_index(nexusindex):
    files = os.listdir('nexus/')
    changes = False
    for file in files:
        if file_exists(file, nexusindex):
            continue
        changes = True
        vector = gpt3_embedding(read_file(memorydir+file))
        nexusindex.append({'filename':file, 'vector': vector})
    return nexusindex


if __name__ == '__main__':
    while True:
        nexusindex = update_index(nexusindex)
        latest = latest_memories(10)
        stacked = stack_memories(latest)
        oldest_time = get_timestamp(latest[0]['filename'])
        relevant = search_index(stacked, nexusindex, count=5, olderthan=oldest_time)
        memories = stack_memories(relevant + latest)
        cognitive_tasks = list()
        immediate_thoughts = list()
        for d in context_prompts:  # orientation and grounding loop
            response = generate_prompt(d[0], memories)
            content = d[1] + response
            print(content)
            save_memory(content, d[2])
            immediate_thoughts.append(content)
        context = '\n'.join(immediate_thoughts)
        context = 'Memories:\n%s\n\nContext:\n%s' % (memories, context)
        for c in control_prompts:
            response = generate_prompt(c[0], context)
            content = c[1] + response
            print(content)
            save_memory(content, c[2])
            #immediate_thoughts.append(content)

        sleep(tempo)