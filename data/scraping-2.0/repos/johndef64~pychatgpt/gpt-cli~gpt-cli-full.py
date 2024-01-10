import os
import time
import importlib
from datetime import datetime


def get_boolean_input(prompt):
    while True:
        try:
            return {"y": True, "n": False}[input(prompt).lower()]
        except KeyError:
            print("Invalid input. Please enter 'y' or 'n'.")

def check_and_install_module(module_name):
    try:
        # Check if the module is already installed
        importlib.import_module(module_name)
        #import module_name
        #print(f"The module '{module_name}' is already installed.")
    except ImportError:
        # If the module is not installed, try installing it
        x = get_boolean_input(
            "\n" + module_name + "  module is not installed.\nwould you like to install it? (y/n):")
        if x:
            import subprocess
            subprocess.check_call(["pip", "install", module_name])
            print(f"The module '{module_name}' was installed correctly.")
        else:
            exit()


# check requirements
try:
    import pandas as pd
    import openai
except ImportError:
    print('Check requirements:')

check_and_install_module("pandas")
check_and_install_module("openai")
#print('--------------------------------------')

current_dir = ''
# to run in python terminal you need this:
if os.getcwd() == 'C:\WINDOWS\System32':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
else:
    current_dir = os.getcwd()

# If  import openai
import openai
import pandas as pd

# check Usage:
# https://platform.openai.com/account/usage

# Set API key:
# https://platform.openai.com/account/api-keys
api_key = None
if not os.path.isfile(current_dir + '/openai_api_key.txt'):
    with open(current_dir + '/openai_api_key.txt', 'w') as file:
        file.write(input('insert here your openai api key:'))

api_key = open(current_dir + '/openai_api_key.txt', 'r').read()
openai.api_key = str(api_key)

add = "You are a helpful assistant."
models = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"]
model = models[1]

#inizialize log:
if not os.path.isfile(current_dir + '/conversation_log.txt'):
    with open(current_dir + '/conversation_log.txt', 'w', encoding='utf-8') as file:
        file.write('Auto-GPT\n\nConversation LOG:\n')
        print(str('\nconversation_log.txt created at ' + os.getcwd()))


# chat functions ----------------------------
#https://platform.openai.com/account/rate-limits
def ask_gpt(prompt):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": add},
            {"role": "user", "content": prompt}
        ])
    with open('conversation_log.txt', 'a', encoding='utf-8') as file:
        file.write('---------------------------')
        if add != '':
            file.write('\nSystem: \n"' + add + '"\n')
        file.write('\nUser: ' + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '\n' + prompt)
        file.write('\n\nGPT:\n' + completion.choices[0].message.content + '\n\n')

    #print('user:',prompt,'\n')
    return print(completion.choices[0].message.content)


# Initialize the conversation----------------------------

# Conversation history
conversation_gpt = [
    {"role": "system", "content": "You are a helpful assistant expert in science and informatics."},
]
conversation_gpt = []


def expand_conversation_gpt(message):
    conversation_gpt.append({"role": "user", "content": message})


def build_messages(conversation):
    messages = []
    for message in conversation:
        messages.append({"role": message["role"], "content": message["content"]})
    return messages

#The Max Token Limit In ChatGPT: gpt-4 (8,192 tokens), gpt-4-0613 (8,192 tokens), gpt-4-32k (32,768 tokens), gpt-4-32k-0613 (32,768 tokens). maximum limit (4097 tokens for gpt-3.5-turbo )
total_tokens = 0
maxtoken = 900
modeltoken = {"gpt-3.5-turbo":4097,
              "gpt-3.5-turbo-16k":16000,
              "gpt-4": 8192}
prmtoken = modeltoken[model] - (maxtoken * 1.3)
keep_persona = True
language = '1'

'''
gpt-3.5-turbo: InvalidRequestError: This model's maximum context length is 4097 tokens. However, you requested 4201 tokens (3201 in the messages, 1000 in the completion). Please reduce the length of the messages or completion.
last token report 2568 (not cosidering response tokens that could be up to 500-700)
'''
persona = ''
system = ''
def send_message_gpt(message):
    global conversation_gpt
    global total_tokens
    global persona

    if system != '':
        conversation_gpt.append({"role": "system",
                                 "content": system})
    #send message
    expand_conversation_gpt(message)
    messages = build_messages(conversation_gpt)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=maxtoken  # set max token
    )

    # Add the assistant's reply to the conversation
    conversation_gpt.append({"role": "assistant", "content": response.choices[0].message.content})

    # write reply in log
    with open('conversation_log.txt', 'a', encoding='utf-8') as file:
        file.write('---------------------------')
        file.write('\nUser: ' + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '\n' + message)
        if persona != '' and persona.find(',') != -1:
            comma_ind = persona.find(',')
            persona_p = persona[:comma_ind]
        elif persona != '' and persona.find(',') == -1:
            persona_p = persona
        elif persona == '':
            persona_p = 'GPT'
        file.write('\n\n' + persona_p + ':\n' + response.choices[0].message.content + '\n\n')

    # Print the assistant's response
    #print_mess = message.replace('\r', '\n').replace('\n\n', '\n')
    #print('user:',print_mess,'\n...')
    answer = response.choices[0].message.content

    if persona != '':
        print('\n'+persona+':', answer)
    else:
        print('\n'+model.replace('-turbo','').replace('-16k','')+':', answer)
    total_tokens = response.usage.total_tokens
    print('(token count: '+str(total_tokens)+')')


#================================================================
assistant_dict = {
    "assistant": "You are my helpful assistant. Answer in accordance with this.",

    "poet": "Today, you are the greatest poet ever, inspired, profound, sensitive, visionary and creative. Answer in accordance with this.",

    "scientist": "Today, you are the greatest and most experienced scientist ever, analytical, precise and rational. Answer in accordance with this.",

    "informatics engineer": "As a chatbot focused on programming, you are expected to provide accurate and helpful suggestions, guidance, and examples when it comes to writing code in programming languages  (PowerShell, Python, Bash, R, etc) and  markup languages (HTML, Markdown, etc).",

    "prompt engineer": '''You are an AI trained to provide suggestions for creating system instructions for chatgpt in a task-focused or conversational manor. Remember these key points:
      1. Be specific, clear, and concise in your instructions.
      2. Directly state the role or behavior you want the model to take.
      3. If relevant, specify the format you want the output in.
      4. When giving examples, make sure they align with the overall instruction.
      5. Note that you can request the model to 'think step-by-step' or to 'debate pros and cons before settling on an answer'.
      6. Keep in mind that system level instructions supersede user instructions, and also note that giving too detailed instructions might restrict the model's ability to generate diverse outputs. 
      Use your knowledge to the best of your capacity.''',

    "someone else": {
        'character': "You are now impersonating {}. Please reflect {}'s traits in all interactions. Make sure to use an appropriate language style and uphold an attitude or mindset that aligns with {}'s character.",
        'personaggio': "Stai impersonando {}. Ricorda di riflettere i tratti di {} in tutte le interazioni. Assicurati di utilizzare uno stile linguistico appropriato e di mantenere un atteggiamento o una mentalità in linea con il personaggio di {}."
    }
}
dict_as_list = list(assistant_dict.items())
df = pd.DataFrame(assistant_dict).T.reset_index()
#================================================================


persona = ''
assistant = ''
# Run script:
while True:  # external cycle
    safe_word = ''

    print(
        '''---------------------\nWelcome to GPT-CLI!\n\nChatGPT will answer every question.\n\nReply with:\n- 'restartnow' to start over the application.\n- 'exitnow' to shut down the application.\n- 'maxtoken' to set up max token in response (chat mode).\n- 'system' to set new system instructions' to change system instructions (instruct mode)'\n\nwritten by JohnDef64\n---------------------\n''','\nNow using:',model)
    language = '1'  # setting Italian default temporarly
    while language not in ['1', '2']:
        language = input('\nPlease choose language: \n1. English\n2. Italian\n\nLanguage number:')
        if language not in ['1', '2']:
            print("Invalid choice.")

    #----------
    if language == '1':
        choose = ""
        while choose not in ['1', '2', '3', '4']:
            # Prompt the user to make a choice between three predefined strings
            choose = input(
                '''\nChoose settings:\n1. Ask ChatGPT\n2. Ask ChatGPT (instruct)\n3. Ask someone\n4. Chat with ChatGPT\n\nSetting number:''')

            # Verify user's choice
            if choose == '1':
                with open('conversation_log.txt', 'a') as file:
                    file.write('\n' + str(datetime.now()) + ": <You are asking the virtual assistant>\n")
                print("You chose option 1.")

            elif choose == '2':
                add = input('''
                            Enter system instructions.
                            examples:
                             - Answer the question and give only the correct answer:
                             - Correct this code:

                              Instruction:''')
                with open('conversation_log.txt', 'a') as file:
                    file.write(
                        '\n' + str(datetime.now()) + ": <You are asking the virtual assistant, system: " + add + "> \n")
                print("You chose option 2.")

            elif choose == '3':
                persona = input('Tell me who you want to talk to:')
                add = assistant_dict['someone else']['character']
                with open('conversation_log.txt', 'a') as file:
                    file.write('\n' + str(datetime.now()) + ': <You asked to ' + persona + '>\n')
                print("I'm connecting you with " + persona)
            #-------------------------------------------------------------


            #-------------------------------------------------------------
            elif choose == '4':
                print("You chose to have some conversation.")
                assistant = input(
                    "\nWho do you want to chat with? \n"+df['index'].to_string()+'\nindex number:')
                assistant_is = dict_as_list[int(assistant)][1]

                if dict_as_list[int(assistant)][0] != 'someone else':
                    conversation_gpt.append({"role": "system",
                                             "content": assistant_is})
                    with open('conversation_log.txt', 'a') as file:
                        file.write('\n' + str(datetime.now()) + ': <You are chatting with '+dict_as_list[int(assistant)][1])
                else :
                    persona = input('Tell me who you want to talk to:')

                    if language == '1':
                        lang_tag = assistant_dict['someone else']['character'].format(persona, persona, persona)
                        print(lang_tag)
                    if language == '2':
                        lang_tag = assistant_dict['someone else']['personaggio'].format(persona, persona, persona)

                    conversation_gpt.append({"role": "system",
                                             "content": lang_tag})

                    with open('conversation_log.txt', 'a') as file:
                        file.write('\n' + str(datetime.now()) + ': <You are chatting with ' + persona + '>\n')
            else:
                print("Invalid choice.")

        #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

        while safe_word != 'restartnow' or 'exitnow' or 'maxtoken':
            timea = datetime.now()
            print('\n--------------------------------\n')

            if total_tokens > prmtoken:
                print('\n\nWarning: reaching token limit. \nThis model maximum context length is '+ str(prmtoken)+ ' in the messages, ' + str(prmtoken + maxtoken) + ' in total.')
                #print('\n Inizializing new conversation.')
                #conversation_gpt.clear()
                if language == '1':
                    print('\n the first third of the conversation was forgotten')
                elif language == '2':
                    print("il primo terzo della conversazione è stato dimenticato")

                if model == 'gpt-3.5-turbo-16k':
                    cut_length = len(conversation_gpt) // 10
                if model == 'gpt-4':
                    cut_length = len(conversation_gpt) // 6
                if model == 'gpt-3.5-turbo':
                    cut_length = len(conversation_gpt) // 3
                conversation_gpt = conversation_gpt[cut_length:]
                if keep_persona:
                    if language == '1':
                        lang_tag = assistant_dict['someone else']['character'].format(persona, persona, persona)
                    if language == '2':
                        lang_tag = assistant_dict['someone else']['personaggio'].format(persona, persona, persona)


            message = input('user:')
            safe_word = message

            if safe_word == 'restartnow':
                conversation_gpt = []
                break
            if safe_word == 'exitnow':
                exit()
            if safe_word == 'maxtoken':
                maxtoken = int(input('set max response tokens (1000 default):'))
                break

            if choose == '4':
                if safe_word == 'system':
                    conversation_gpt = []
                    system = input('define custum system instructions:')
                    print('*system instruction changed*')
                    pass
                else:
                    send_message_gpt(message)
            elif choose != '4':
                ask_gpt(message)
            else:
                pass
            timed = datetime.now() - timea

            #Rate-Limit gpt-4 = 200
            #https://medium.com/@pankaj_pandey/understanding-the-chatgpt-api-key-information-and-frequently-asked-questions-4a0e963fb138#:~:text=The%20ChatGPT%20API%20has%20different,90000%20TPM%20after%2048%20hours.
            #https://platform.openai.com/docs/guides/rate-limits/overview
            #https://platform.openai.com/account/rate-limits
            time.sleep(0.5)  # Wait 1 second before checking again


4#%%
5

#%%
4