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
    import pyperclip
    import openai
except ImportError:
    print('Check requirements:')
check_and_install_module("pyperclip")
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
import pyperclip

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


# Log Inizialization:
if not os.path.isfile(current_dir + '/conversation_log.txt'):
    with open(current_dir + '/conversation_log.txt', 'w') as file:
        file.write('Auto-GPT\n\nConversation LOG:\n')
        print(str('\nconversation_log.txt created at ' + os.getcwd()))

# Parameters--------------------------
total_tokens = 0
maxtoken = 1000
prmtoken = 8100 - (maxtoken*1.3)
keep_persona = True
persona = ''
model = 'gpt-4'
add = "You are a helpful assistant."

'''InvalidRequestError: This model's maximum context length is 4097 tokens. However, you requested 4201 tokens (3201 in the messages, 1000 in the completion). Please reduce the length of the messages or completion.
last token report 2568 (not cosidering response tokens that could be up to 500-700)
'''


# Applications dictionaries
dicteng = {
    'title': "---------------------\nWelcome to Auto-GPT!\n\nGPT will answer everything you will send to clipboard.\n\nSend to clipboard:\n- 'restartnow' to start over the application.\n- 'exitnow' to shut down the application.\n- 'maxtoken' to set up max token in response (chat mode).\n- 'rest'/'wake' To temporarily suspend and wakeup the application.\n- 'sys: (new system instructions)' to change system instructions (instruct mode)'\n\nwritten by JohnDef64\n---------------------\n",

    'language': '\nPlease choose language: \n1. English\n2. Italian\n\nLanguage number:',

    'chat': "\nChoose chat-mode: \n1. Ask to gpt-4\n2. Ask to gpt-4 (instruct)\n3. Ask to someone\n4. Chat with gpt-4\n\nNumber:",

    'instruct': '''\nEnter system instructions.\nexamples:\n - Answer the question and give only the correct answer:\n - Correct this code:\n  Instruction:''',

    'instruction': "\nWho do you want to converse with?: \n1. the assistant\n2. the poet \n3. the scientist\n4. someone else\n\nNumber: ",

    'instructions': {'assistant': "Today, you are the 'expert assistant in everything. Answer in accordance with this.",
                     'poet': "Today, you are the greatest poet ever, inspired, profound, sensitive, visionary and creative. Answer in accordance with this.",
                     'scientist': "Today, you are the greatest and most experienced scientist ever, analytical, precise and rational. Answer in accordance with this.",
                     'tell me': 'Tell me who you want to talk to:'
                     }
}

dictita = {
    'title': "---------------------\nBenvenuto in Auto-GPT!\n\nGPT risponderà a tutto ciò che invierai nella clipboard.\n\nInvia alla clipboard:\n- 'restartnow' per riavviare l'applicazione.\n- 'exitnow' per spegnere l'applicazione.\n- 'maxtoken' per impostare il massimo numero di token nella risposta (modalità chat).\n- 'rest'/'wake' Per sospendere temporaneamente e risvegliare l'applicazione.\n- 'sys: (nuove istruzioni di sistema)' per cambiare le istruzioni di sistema (modalità istruzione)'\n\nscritto da JohnDef64\n---------------------\n",

    'language': '\nPer favore scegli la lingua: \n1. Inglese\n2. Italiano\n\nNumero della lingua:',

    'chat': "\nScegli la modalità chat: \n1. Chiedi a gpt-4\n2. Chiedi a gpt-4 (istruzione)\n3. Chiedi a qualcuno\n4. Chatta con gpt-4\n\nNumero:",

    'instruct': '''\nInserisci le istruzioni di sistema.\nesempi:\n - Rispondi alla domanda e dai solo la risposta corretta:\n - Correggi questo codice:\n  Istruzione:''',

    'instruction': "\nCon chi vuoi conversare?: \n1. l'assistente\n2. il poeta \n3. lo scienziato\n4. qualcun altro\n\nNumero: ",

    'instructions': {'assistant': "Oggi, tu sei l' 'esperto assistente in tutto. Rispondi in conformità con questo.",
                     'poet': "Oggi, tu sei il più grande poeta di sempre, ispirato, profondo, sensibile, visionario e creativo. Rispondi in conformità con questo.",
                     'scientist': "Oggi, tu sei il più grande e più esperto scienziato di sempre, analitico, preciso e razionale. Rispondi in conformità con questo.",
                     'tell me': 'Dimmi con chi vuoi parlare:'
                     }
}

language = ''
dict = dicteng


# Chat Functions ----------------------------
#https://platform.openai.com/account/rate-limits
def ask_gpt(prompt):
    completion = openai.ChatCompletion.create(
        model= model,
        messages=[
            {"role": "system", "content": add},
            {"role": "user", "content": prompt}
        ])

    with open('conversation_log.txt', 'a', encoding='utf-8') as file:
        file.write('---------------------------')
        if add != '':
            file.write('\nSystem: \n"' + add+'"\n')
        file.write('\nUser: '+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n' + prompt)
        file.write('\n\nGPT:\n' + completion.choices[0].message.content + '\n\n')

    print_mess = prompt.replace('\r', '\n').replace('\n\n', '\n')
    print('user:',print_mess,'\n')
    return print(completion.choices[0].message.content)

# Initialize the conversation (text-davinci-003)----------------
conversation = [
    {"role": "system", "content": "You are a helpful assistant expert in science and informatics."},
]
def expand_conversation(message):
    conversation.append({'role': 'user', 'content': message})
def build_prompt(conversation):
    prompt = ""
    for message in conversation:
        role = message['role']
        content = message['content']
        prompt += f'{role}: {content}\n'
    return prompt
def send_message(message):
    expand_conversation(message)
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=build_prompt(conversation),
        temperature=0.7,
        max_tokens=2000,
        n=1,
        stop=None
    )
    conversation.append({'role': 'assistant', 'content': response.choices[0].text.strip()})
    # Print the assistant's response
    print(response.choices[0].text.strip())


# Initialize the conversation (gpt-4)----------------------------

# Conversation history
conversation_gpt = []

def expand_conversation_gpt(message):
    conversation_gpt.append({"role": "user", "content": message})

def build_messages(conversation):
    messages = []
    for message in conversation:
        messages.append({"role": message["role"], "content": message["content"]})
    return messages


def send_message_gpt(message):
    global conversation_gpt
    global total_tokens

    if total_tokens > prmtoken:
        print('\n\nWarning: reaching token limit. \nThis model maximum context length is ', prmtoken, ' in the messages,', prmtoken+maxtoken ,'in total.')
        #print('\n Inizializing new conversation.')
        #conversation_gpt.clear()
        print('\n the first third of the conversation was forgotten')

        quarter_length = len(conversation_gpt) // 3
        conversation_gpt = conversation_gpt[quarter_length:]
        if keep_persona:
            conversation_gpt.append({"role": "system",
                                     "content": dict['instructions']['persona']})

    expand_conversation_gpt(message)
    messages = build_messages(conversation_gpt)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=maxtoken  # set max token
    )
    # Add the assistant's reply to the conversation
    with open('conversation_log.txt', 'a', encoding='utf-8') as file:
        file.write('---------------------------')
        file.write('\nUser: '+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n' + message)
        if persona != '':
            file.write('\n\n'+persona+':\n' + response.choices[0].message.content + '\n\n')
        else:
            file.write('\n\nGPT:\n' + response.choices[0].message.content + '\n\n')
    conversation_gpt.append({"role": "assistant", "content": response.choices[0].message.content})

    # Print the assistant's response
    print_mess = message.replace('\r', '\n').replace('\n\n', '\n')
    print('user:',print_mess,'\n...')
    print(response.choices[0].message.content)

    total_tokens = response.usage.total_tokens
    print('prompt tokens:', total_tokens)


#-----------------------------------------------------
# Run script:
while True:  # external cycle
    safe_word = ''
    pyperclip.copy('')
    print(dict['title'])

    while language not in ['1', '2']:
        language = input(dict['language'])
        if language not in ['1', '2']:
            print("Invalid choice.")
    if language == '1':
        dict = dicteng
    else:
        dict = dictita
    #----------
    #if language == '1':
    choose = ""
    while choose not in ['1', '2', '3', '4']:
        # Prompt the user to make a choice between three predefined strings
        choose = input(dict['chat'])

        # Verify user's choice
        if choose == '1':
            with open('conversation_log.txt', 'a') as file:
                file.write('\n' + str(datetime.now()) + ": <You are asking the virtual assistant>\n")
            print("You chose option 1.")

        elif choose == '2':
            add = input(dict['instruct'])
            with open('conversation_log.txt', 'a') as file:
                file.write('\n' + str(datetime.now()) + ": <You are asking the virtual assistant, system: "+add+"> \n")
            print("You chose option 2.")

        elif choose == '3':
            persona = input('Tell me who you want to talk to:')
            persona_dict = {'character': "You are now impersonating "+persona+". Please reflect "+persona+"'s traits in all interactions. Make sure to use an appropriate language style and uphold an attitude or mindset that aligns with "+persona+"'s character.",
                            'personaggio': "Stai impersonando "+persona+", . Ricorda di riflettere i tratti di "+persona+" in tutte le interazioni. Assicurati di utilizzare uno stile linguistico appropriato e di mantenere un atteggiamento o una mentalità in linea con il personaggio di "+persona}
            add = persona_dict['character']
            with open('conversation_log.txt', 'a') as file:
                file.write('\n' + str(datetime.now()) + ': <You asked to ' + persona + '>\n')
            print("I'm connecting you with " + persona)

        elif choose == '4':
            print("You chose to have some conversation.")
            assistant = input(dict['instruction'])
            if assistant == '1':
                conversation_gpt.append({"role": "system",
                                         "content": dict['instructions']['assistant']})
                with open('conversation_log.txt', 'a') as file:
                    file.write('\n' + str(datetime.now()) + ': <You are chatting with the general assistant>\n')
            elif assistant == '2':
                conversation_gpt.append({"role": "system",
                                         "content": dict['instructions']['poet']})
                with open('conversation_log.txt', 'a') as file:
                    file.write('\n' + str(datetime.now()) + ': <You are chatting with the greatest poet ever>\n')
            elif assistant == '3':
                conversation_gpt.append({"role": "system",
                                         "content": dict['instructions']['scientist']})
                with open('conversation_log.txt', 'a') as file:
                    file.write(
                        '\n' + str(datetime.now()) + ': <You are chatting with the greatest  scientist ever>\n')
            elif assistant == '4':
                persona = input(dict['instructions']['tell me'])
                persona_dict = {'character': "You are now impersonating "+persona+". Please reflect "+persona+"'s traits in all interactions. Make sure to use an appropriate language style and uphold an attitude or mindset that aligns with "+persona+"'s character.",
                                'personaggio': "Stai impersonando "+persona+", . Ricorda di riflettere i tratti di "+persona+" in tutte le interazioni. Assicurati di utilizzare uno stile linguistico appropriato e di mantenere un atteggiamento o una mentalità in linea con il personaggio di "+persona}
                if language == '1':
                    content = persona_dict['character']
                elif language == '2':
                    content = persona_dict['personaggio']
                conversation_gpt.append({"role": "system",
                                         "content": content})
                with open('conversation_log.txt', 'a') as file:
                    file.write('\n' + str(datetime.now()) + ': <You are chatting with ' + persona + '>\n')
        else:
            print("Invalid choice.")

    previous_content = ''  # Initializes the previous contents of the clipboard

    while safe_word != 'restartnow' or 'exitnow' or 'maxtoken' or 'wake':
        safe_word = pyperclip.paste()
        clipboard_content = pyperclip.paste()  # Copy content from the clipboard

        x = True
        while safe_word == 'rest':
            if x : print('<sleep>')
            x = False
            clipboard_content = pyperclip.paste()
            time.sleep(1)
            if clipboard_content == 'wake':
                print('<Go!>')
                break

        if clipboard_content != previous_content and clipboard_content.startswith('sys:'):
            add = clipboard_content.replace('sys:','')
            previous_content = clipboard_content
            print('\n<system setting changed>\n')

        if clipboard_content != previous_content and clipboard_content != 'restartnow':  # Check if the content has changed
            #print(clipboard_content)  # Print the content if it has changed
            print('\n\n--------------------------------\n')
            previous_content = clipboard_content  # Update previous content

            if choose == '4':
                send_message_gpt(clipboard_content)

            else:
                ask_gpt(clipboard_content)

        time.sleep(1)  # Wait 1 second before checking again
        if safe_word == 'restartnow':
            break

        if safe_word == 'exitnow':
            exit()

        if safe_word == 'maxtoken':
            #global maxtoken
            #custom = get_boolean_input('\ndo you want to cutomize token options? (yes:1/no:0):')
            #if custom:
            maxtoken = int(input('set max response tokens (800 default):'))
            break


# openai.error.InvalidRequestError: This model's maximum context length is 4097 tokens.  However, you requested 4116 tokens (2116 in the messages, 2000 in the completion).  Please reduce the length of the messages or completion.