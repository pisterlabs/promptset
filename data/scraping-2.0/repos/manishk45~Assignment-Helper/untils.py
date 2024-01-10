import os


from database_api import save_message_to_db_perm,save_message_to_db_temp,get_previous_messages_temp,del_previous_conversations_temp
import openai
from dotenv import load_dotenv
load_dotenv()



openai.api_key = os.getenv('OPENAI_API_KEY')

#function for fomatting meesgae suitable for openai
def format_messages(chat_history: list[list]) -> list[dict]:
    formated_messages = [
        {"role": "system", "content": "You are a helpful, creative, and clever assistant"}
    ]
    for i in range(len(chat_history)):
        ch = chat_history[i]
        formated_messages.append(
            {
                "role": "user",
                "content": ch[0]
            }
        )
        if ch[1] != None:
            formated_messages.append(
                {
                    "role": "assistant",
                    "content": ch[1]
                }
            )
    return formated_messages


#function for counting tokens 
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_response(text: str, chatbot: list[list],participant_id: int, participant_mail: str) -> tuple:
    #list of participants
    participant_list=[21018]
    part_id=int(participant_id)
    if part_id in participant_list:
        if len(chatbot)==1:
            previous_conversation = get_previous_messages_temp(participant_id)
            previous_conversation.extend(chatbot) #chatbot content will be appended in the last of previous conversations
        else:
            previous_conversation=chatbot

        previous_conversation[-1][1] = ''
        previous_conversation[-1][1] = ''
        formated_messages = format_messages(previous_conversation)
        try:
            response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo-16k',
                        messages=formated_messages,
                        stream=True
                        )
            for chunk in response:
                delta = chunk['choices'][0]['delta']
                if 'content' in delta.keys():
                    previous_conversation[-1][1]+= delta['content'] #emptied the last none and added the generated content which would streamed to the caller
                    yield previous_conversation
        except:
            error_message = "We are facing a technical issue at this moment."
            previous_conversation[-1][1] = ""
            for character in error_message:
                previous_conversation[-1][1] += character
                yield previous_conversation
        tokens_in_prompt= num_tokens_from_string(previous_conversation[-1][0])
        tokens_in_response= num_tokens_from_string(previous_conversation[-1][1])
        tokens_used= tokens_in_prompt +tokens_in_response
        save_message_to_db_perm(previous_conversation[-1][0],previous_conversation[-1][1],participant_id,participant_mail)
        save_message_to_db_temp(previous_conversation[-1][0],previous_conversation[-1][1],tokens_used,participant_id,participant_mail)
        return previous_conversation
    else:
        id_error_message = "Sorry, Your Id is Wrong"
        chatbot[-1][1] = ""
        for character in id_error_message:
            chatbot[-1][1] += character
            yield chatbot
        return chatbot

def set_user_query(text: str, chatbot: list[list]) -> tuple:
    #print(chatbot)
    chatbot.append([text, None])
    return '', chatbot


def clear_conversations(participant_id: int, participant_mail: str) -> tuple:
    del_previous_conversations_temp(participant_id)
    return '', []

def regenerate_response(chatbot: list[list],participant_id: int, participant_mail: str) -> tuple:
    #print(int(participant_id))
    text=chatbot[-1][0]
    chatbot.pop()
    chatbot.append([text, None])
    previous_conversation=chatbot
    previous_conversation[-1][1] = ''
    formated_messages = format_messages(previous_conversation)
    try:
        response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo-16k',
                    messages=formated_messages,
                    stream=True
                    )
        for chunk in response:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta.keys():
                previous_conversation[-1][1]+= delta['content'] #emptied the last none and added the generated content which would streamed to the caller
                yield previous_conversation
    except:
        error_message = "We are facing a technical issue at this moment."
        previous_conversation[-1][1] = ""
        for character in error_message:
            previous_conversation[-1][1] += character
            yield previous_conversation
    
    return previous_conversation






'''
def regenerate_response(text: str, chatbot: list[list],participant_id: int, participant_mail: str) -> tuple:
    #print(int(participant_id))
    text=chatbot[-1][0]
    chatbot.pop()
    chatbot.append([text, None])
    chat_regenerated= generate_response(text,chatbot,participant_id, participant_mail)
    return chat_regenerated

    def regenerate_response(chatbot: list[list],participant_id: int, participant_mail: str) -> tuple:
    #print(int(participant_id))
    text=chatbot[-1][0]
    chatbot.pop()
    chatbot.append([text, None])
    chat_regenerated= generate_response(text,chatbot,participant_id, participant_mail)
    return chat_regenerated
'''
