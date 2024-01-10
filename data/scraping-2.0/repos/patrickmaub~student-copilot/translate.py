import csv
import openai
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')


default_prompt = "Translate the following text to {}."


def translation(user_prompt, instructions):

    summaries = []

    i = 0
    b = 4000
    while i < len(user_prompt):

        summaries.append('Translate the text below to ' +
                         instructions + '\n\Text to translate:\n')
        summaries[-1] += (user_prompt[i:b])
        i += 4000
        b += 4000

    for i in range(len(summaries)):

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=summaries[i],
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        print(response['choices'][0]['text'])
    return(response['choices'][0]['text'])


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

def add_ons(user_prompt):
    print('What language would you like to translate to?')
    print('1. Russian')
    print('2. German')
    print('3. French')
    print('4. Chinese')
    print('5. English')
    print('6. Catalan')
    print('7. Introductory Level')
    print('7. Intermediate Level Level')

    instructions = []
    user_choice = input()

    if '1' in user_choice:
        instructions.append(
            'Russian')
    if '2' in user_choice:
        instructions.append(
            'German')
    if '3' in user_choice:
        instructions.append(
            'French')

    if '4' in user_choice:
        instructions.append(
            'Chinese')
    if '5' in user_choice:
        instructions.append(
            'English')
    if '6' in user_choice:
        instructions.append(
            'Catalan')
    if '7' in user_choice:
        instructions.append(
            'The grammar should be flawless, but the vocabulary should be that of a beginner who is new to the language.')
    if '8' in user_choice:
        instructions.append(
            'The grammar should be flawless, but the vocabulary should be that of a moderate speaker of the language who has been studying for 2 years.')

    txt = ''
    for i in instructions:
        txt += i + '\n'
    print('\n' + txt + '\n')
    print(translation(user_prompt, txt))
    # make it so that they can add multiple add ons at once


def read_txt_file():
    print("Python Program to print list the files in a directory.")
    Direc = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(Direc)
    files = [f for f in files if os.path.isfile(Direc+'/'+f)]
    for i in range(len(files)):
        print(i+1, files[i])
    print('Please enter the name of the file you would like to translate.')
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


def translate():
    print('Welcome to the AI text translater. Would you like to input the text or pick a file?')
    print('1. Input text')
    print('2. Pick a file')
    user_choice = input()
    if user_choice == '1':
        print('Please enter the text you would like to translate.')
        user_prompt = input()
        add_ons(user_prompt)
    elif user_choice == '2':
        user_prompt = read_txt_file()
        add_ons(user_prompt)

    else:
        print('Please enter a valid choice.')
        translate()
