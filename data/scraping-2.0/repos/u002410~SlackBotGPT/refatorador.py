
import openai
from secret_access import OPEN_IA_TOKEN
from filter_pii import remove_pii, contains_prohibited

openai.api_key = OPEN_IA_TOKEN

conversations = {}
message_start_ts = {}
info = {}

def process_code(message, say, context_type='random'):
    user_id = message['user']

    user_message = ""
    if context_type == 'refactor':
        user_message = f"I need your help with a piece of code in {info[user_id]['language']}. Here is the code:\n{info[user_id]['code']}.\n"
        if 'alteration' in info[user_id]:
            user_message += f"The desired change is: {info[user_id]['alteration']}.\nPlease, refactor the code considering this request."
        system_content = "You are a helpful assistant that review and refactor code."
    elif context_type == 'security':
        user_message = f"I need your help with a piece of code. It's written in {info[user_id]['language']} and has a known vulnerability {info[user_id]['vulnerability']}.\nHere is the code:\n\n{info[user_id]['code']}\n\nPlease, refactor this code to address the identified vulnerability and show the lines where the code presents the issue."
        if 'alteration' in info[user_id]:
            user_message += f"The desired change is: {info[user_id]['alteration']}.\nPlease, refactor the code considering this request."
        system_content = "You are a helpful assistant that review and refactor insecure code."
    else: # assuming 'random' context
        user_message = f"Hello assistant, {info[user_id]['question']}."
        system_content = "You are a helpful assistant."

    thread_id = message['ts']

    if thread_id not in conversations:
        conversations[thread_id] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": remove_pii(user_message)}
        ]
        message_start_ts[thread_id] = message['ts']
    else:
        conversations[thread_id].append({"role": "user", "content": user_message})

    message_start_ts[thread_id] = message['ts']

    valid_sensetive = contains_prohibited(user_message)

    if valid_sensetive == user_message:
        response_message = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversations[thread_id],
            max_tokens=1024,
            temperature=0.2,
            top_p = 0
        )

        conversations[thread_id].append(
            {"role": "assistant", "content": response_message['choices'][0]['message']['content']}
            )
        
        # save_conversation(thread_id, conversations[thread_id])
        if context_type !='random':
            if 'ts' in message:
                say(thread_ts=message['ts'], text=response_message['choices'][0]['message']['content'])
                say("Was the refactoring satisfactory? Answer with *Yes* or *No*.", thread_ts=message['ts'])
                info[user_id]['satisfied'] = True
            else:
                if 'ts' in message:
                    # Sempre use o timestamp da mensagem original para responder na mesma thread
                    say(thread_ts=message['ts'], text=response_message['choices'][0]['message']['content'])            
    else:
        say(valid_sensetive, thread_ts=message['ts'])


def process_message(message, say):
    user_id = message['user']
    user_message = message['text']
    thread_id = info[user_id]['thread']

    if thread_id not in conversations:
        system_content = "You are a helpful assistant."
        conversations[thread_id] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": remove_pii(user_message)}
        ]
    else:
        conversations[thread_id].append({"role": "user", "content": remove_pii(user_message)})

    valid_sensetive = contains_prohibited(user_message)

    if valid_sensetive == user_message:
        response_message = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversations[thread_id],
            max_tokens=512,
            temperature=0.2,
            top_p = 0
        )

        conversations[thread_id].append(
            {"role": "assistant", "content": response_message['choices'][0]['message']['content']}
        )

        if 'ts' in message:
            # Sempre use o timestamp da mensagem original para responder na mesma thread
            say(thread_ts=message['ts'], text=response_message['choices'][0]['message']['content'])
    else:
        say(valid_sensetive, thread_ts=message['ts'])