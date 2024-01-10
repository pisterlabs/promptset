import csv
import openai
import os

openai.api_key = os.environ.get('OPENAI_API_KEY')


def make_study_guide(user_prompt, instructions):

    summaries = []

    i = 0
    b = 8000
    while i < len(user_prompt):

        summaries.append('Write a summary for the paper below while accounting for the following instructions:\n' +
                         instructions + '\n\nPaper:\n')
        summaries[-1] += (user_prompt[i:b])
        i += 8000
        b += 8000

    big_summary = ''
    for i in range(len(summaries)):

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=summaries[i],
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        big_summary += response['choices'][0]['text']
        print(response['choices'][0]['text'])

    return big_summary


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
    print('How would you like your study guide made?')
    print('1. Define key terms')
    print('2. Paragraph')
    print('3. Bullet Point')
    print('4. Add Practice Exam Questions')
    print('5. Descriptive')
    print('6. Simple')
    instructions = []
    user_choice = input()

    if '1' in user_choice:
        instructions.append(
            'The study guide should include all key definitions mentioned in the lecture content.')
        print('Define key terms')
    if '2' in user_choice:
        instructions.append(
            'The study guide should be written in paragraph format.')
        print('Long')
    if '3' in user_choice:
        instructions.append(
            'Ensure that the study guide is written in bullet points.')
        print('Short')
    if '4' in user_choice:
        instructions.append(
            'Throughout the study guide, add practice exam questions. Ensure that the questions are written in a way that they can be answered with a single sentence.')
        print('Use Quotes')
    if '5' in user_choice:
        instructions.append(
            'Ensure to make the study guide as descriptive as possible.')
        print('Ensure to summarize any counter arguments made in the paper.')
    if '6' in user_choice:
        instructions.append(
            'Ensure that the study guide is as simple and concise as possible.')
        print('Bullet Points')

    txt = ''
    for i in instructions:
        txt += i + '\n'
    print('\n' + txt + '\n')
    print(make_study_guide(user_prompt, txt))
    # make it so that they can add multiple add ons at once


def read_txt_file():
    print("Python Program to print list the files in a directory.")
    Direc = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(Direc)
    files = [f for f in files if os.path.isfile(Direc+'/'+f)]
    for i in range(len(files)):
        print(i+1, files[i])
    print('Please enter the name of the file you would like to make a study guide from.')
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


def study_guide():
    print('Welcome to the AI study guide maker. Would you like to input the text or pick a file?')
    print('1. Input text')
    print('2. Pick a file')
    user_choice = input()
    if user_choice == '1':
        print('Please enter the text you would like to summarize.')
        user_prompt = input()
        add_ons(user_prompt)
    elif user_choice == '2':
        user_prompt = read_txt_file()
        add_ons(user_prompt)

    else:
        print('Please enter a valid choice.')
        make_study_guide()
