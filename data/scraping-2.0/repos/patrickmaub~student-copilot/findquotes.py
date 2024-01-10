import csv
import random
import openai
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')


def quoter(topic, source, instructions):
    # break into random subchunks of paper. 7000 characters each. keep doing until number of quotes is reached.
    chunks = []

    i = 0
    b = 7000
    while i < len(source + topic):

        chunks.append('You are given the following prompt:' + topic + '\nFind all quotations from the paper below that are relevant to the prompt above while accounting for the following instructions:\nThe quotes should be listed by number, i.e. 1. "Quote 1 Here"\n2."Quote 2 Here", etc.\n' +
                      instructions + '\n\nPaper:\n')
        chunks[-1] += (source[i:b])
        i += 7000
        b += 7000
    all_quotes = ''
    random.shuffle(chunks)
    for i in range(len(chunks)):
        if(len(all_quotes)) > 8000:
            break
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=chunks[i],
            temperature=0.7,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        all_quotes += response['choices'][0]['text']
        print(response['choices'][0]['text'])

        # print(response['choices'][0]['text'])
    return(all_quotes)


# select which add ons to use from a  list.
# add ons are:
# Define key terms
# Long
# Short
# Use Quotes
# Provide Counterargument
# Provide Examples
# Provide References
# Provide Citations

def add_ons(source, topic):
    print('What preferences would you like to find your quotes under?')
    print('1. Short quotes')
    print('2. Long quotes')
    print('3. Rephrased Quotes')
    print('4. Direct Quotes')
    print('5. Explain Relevancy')
    instructions = []
    user_choice = input()

    if '1' in user_choice:
        instructions.append(
            'Ensure to select quotes from the paper that are highly relevant to the topic that are no longer than a couple sentences long.')

    if '2' in user_choice:
        instructions.append(
            'Ensure to select quotes from the paper that are highly relevant to the topic.')
        print('Long')
    if '3' in user_choice:
        instructions.append(
            'Ensure to rephrase the quotes in your own words, while retaining the meaning of the quote.')
        print('Short')
    if '4' in user_choice:
        instructions.append(
            'Ensure that the quotes are direct copies from the paper.')
        print('Use Quotes')
    if '5' in user_choice:
        instructions.append(
            'Under each quote, ensure to explain the relevancy of the quote to the topic in one sentence.')
        print('Ensure to summarize any counter arguments made in the paper.')

    txt = ''
    for i in instructions:
        txt += i + '\n'
    print('\n' + txt + '\n')
    print(quoter(topic, source,  txt))
    # make it so that they can add multiple add ons at once


def read_txt_file():
    print("Python Program to print list the files in a directory.")
    Direc = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(Direc)
    files = [f for f in files if os.path.isfile(Direc+'/'+f)]
    for i in range(len(files)):
        print(i+1, files[i])
    user_choice = input()
    try:
        user_choice = int(user_choice)
        if user_choice > len(files):
            print('Please enter a valid choice.')
            read_txt_file()
        user_choice = files[user_choice-1]
    except:
        print('Please enter a valid choice.')
        if user_choice not in files:
            read_txt_file()
    with open(user_choice, encoding='utf8') as f:
        text = f.read()

    return text


def quotes():
    print('Welcome to the AI quote finder. Would you like to input the text or pick a file?')
    print('1. Input text')
    print('2. Pick a file')
    source = ''
    topic = ''
    user_choice = input()
    if user_choice == '1':
        print('Please enter the text from which you would like to extract quotes.')
        source = input()
        # add_ons(user_prompt)
    elif user_choice == '2':
        print('Select a source from which you would like to extract quotes')
        source = read_txt_file()

    else:
        print('Please enter a valid choice.')
        quotes()

    user_topic = input()
    if user_topic == '1':
        print('Please enter the topic that you would like to find quotes about.')
        topic = input()
        # add_ons(user_prompt)
    elif user_choice == '2':
        topic = read_txt_file()

    else:
        print('Please enter a valid choice.')
        quotes()

    add_ons(source, topic)
