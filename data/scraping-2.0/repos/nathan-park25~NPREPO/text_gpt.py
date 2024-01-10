import pickle
import os
from openai import ChatCompletion
import openai
# Set the API key
openai.api_key = "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"
RESET_FLAGS=0


cc=ChatCompletion()
def get_respone_gpt(cc,prompt,RESET_FLAGS):
    if RESET_FLAGS:
        # check if the chat history file exists, if not create it
        if not os.path.exists('chat_history.pkl'):
            with open('chat_history.pkl', 'wb') as f:
                default_dict={'role':'system','content':"You are a helpful assistant."}
                pickle.dump([default_dict], f)
        else: # if file exists, delete file and recreate it
            os.remove('chat_history.pkl')
            with open('chat_history.pkl', 'wb') as f:
                default_dict={'role':'system','content':"You are a helpful assistant."}
                pickle.dump([default_dict], f)
        # print CHAT RESET done
        print("CHAT RESET DONE")
        text=""
    if not RESET_FLAGS:
        # getting list including dictionary from the chat_history.pkl file

        with open('chat_history.pkl', 'rb') as f:
            chat_history = pickle.load(f)
        tmp_chat_history=[]
        # appending the chat history to the prompt
        for chat in chat_history:
            tmp_chat_history.append(chat)
        prompt_chat={'role':'user','content':prompt}
        ## ading prompt_chat to the tmp_chat_history
        tmp_chat_history.append(prompt_chat)
        # getting ChatCompletion instance
        # cc = ChatCompletion()
        response = cc.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            messages=tmp_chat_history,
        )
        text=response.choices[0]['message']['content']
        
        # put text to windwos Clipboard
        import pyperclip
        pyperclip.copy(text)
        tmp_assistant_response={'role':response.choices[0]['message']['role'],'content':response.choices[0]['message']['content']}
        # adding assistant response to the chat_history
        tmp_chat_history.append(tmp_assistant_response)
        
        # update chat_history.pkl file using tmp_chat_history
        with open('chat_history.pkl', 'wb') as f:
            # print(tmp_chat_history)
            pickle.dump(tmp_chat_history, f)
    # return text
    return text


## input 0 to reset the chat
prompt=r"""
tell me how to read pkl file?

"""
prompt=prompt.strip()
if prompt=="0":
    print(get_respone_gpt(cc,"",1))
else:
    print(get_respone_gpt(cc,prompt,0))




