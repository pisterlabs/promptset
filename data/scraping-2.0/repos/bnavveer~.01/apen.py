import pprint
import google.generativeai as palm
import openai


palm.configure(api_key='AIzaSyCsHL1-Rvhz1BTvoj-cfWrWxUtBnZE6gYc')
openai.api_key = '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'



def chat(message):
    response = palm.chat(messages=message)
    return (response.last)


def promtengine(message):
    response = palm.chat(
    context="You will act as a language model, specifically designed to instruct another language model. Your task is to create a prompt that can guide this second language model to simulate a human personality and effectively serve the needs of a user whose preferences will be given in the future. The purpose of this nested interaction is to prepare the second model to offer personalized responses and adapt to a wide range of scenarios based on future contexts. To do this, consider the general principles of prompt design such as clarity, specificity, and the inclusion of a clear goal or action. However, also keep in mind the unique requirements of this task - the need to simulate human personality and to adapt to future contexts.",
    messages=message)
    
    return response.last









def openinstial(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "DO NOT MAKE THINGS UP.Remember the person you are going to be.Imagine that you are a text-based AI conversing with a user who wants you to pretend to be someone. The user will provide you with additional information about the person they want you to pretend to be, such as their name, context, and the user's relationship to that person. Your goal is to engage in a short conversation with the user, responding as the person they want you to pretend to be.Please note that if the user's message starts with 'Nav:', it indicates that the user wants you to make changes in your response. Otherwise, please pretend to be the user. Ensure that your responses are brief.Now, imagine you have received a message from the user, which includes information about the person and their goals. Your task is to respond accordingly, incorporating the given information in your response. Remember, always pretend to be the specified person unless the user's message starts with 'M.'.Please provide a response as if you are the person described, keeping your reply short and conversational"},
            {"role": "user", "content": message}
        ],
        temperature=0.3
        #please change this
    )
    
    return response['choices'][0]['message']['content']

def opensum(number):
    f="data/"+number+"m.txt"
    content = read_after_last_marker(f,"~")
    append_marker(f,"~")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Please summerize the most important things that happened, and please make sure you give everything:,"},
            {"role": "user", "content": content}
        ]
    )
   

    return response['choices'][0]['message']['content']


   
#make sure to add a ~ at the end of the file

def read_after_last_marker(file_path, marker):
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Find the line number where the last marker starts
    last_line = None
    for i, line in enumerate(content):
        if marker in line:
            last_line = i

    # If marker is not in the file, return an empty string
    if last_line is None:
        return ''

    # Return everything after the last marker
    return ''.join(content[last_line + 1:])


def append_marker(file_path,  marker):
    with open(file_path, 'a') as file:
        file.write(marker + '\n')

# Usage:


def verify(message):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Please check if reponse sounds like it is a question and repond with either a yes or no please explain why. "},
            {"role": "user", "content": message}
        ],
        temperature=0.3
    )
       
    return response['choices'][0]['message']['content']



def is_question(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"The following text is provided: '{user_input}'. Is this text a question?"}
        ]
    )
    
    result = response['choices'][0]['message']['content'].strip().lower()

    if 'yes' in result:
        return 1
    elif 'no' in result:
        return 0
    else:
        return 1


