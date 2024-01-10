import openai
import argparse

#Streaming endpoint
API_URL = "https://api.openai.com/v1/chat/completions" #os.getenv("API_URL") + "/generate_stream"

# Load your API key from an environment variable or secret management service
openai.api_key_path = "OPENAI_API_KEY.txt"

def predict(input, chat_history, params): 

    # add user input
    new_message = {"role": "user", "content": input}
    chat_history.append(new_message)

    #messages
    payload = {
    "model": "gpt-3.5-turbo",
    "messages": chat_history,
    "temperature" : params.temperature,
    "top_p" : params.top_p,
    "n" : params.n, 
    "stop": params.stop, # eg 4 
    "max_tokens": params.max_tokens, #0-4096
    "presence_penalty": params.presence_penalty,
    "frequency_penalty": params.frequency_penalty, 
    # "logit_bias": None
    }

    response = openai.ChatCompletion.create(**payload)

    reply = response["choices"][0]["message"]["content"]
    tokens_num = response["usage"]
    finish_reason = response["choices"][0]["finish_reason"]

    # add ChatGPT reply
    new_message = {"role": "assistant", "content": reply}
    chat_history.append(new_message)

    return reply, tokens_num, finish_reason, chat_history


def chat(params):

    chat_history=[]
    if params.use_system:
        new_message = {"role": "system", "content": params.system_message}
        print("System: ", params.system_message)
        chat_history.append(new_message)

    while True:
        user_input = input("User: ")
        reply, tokens_num, finish_reason, chat_history = predict(user_input, chat_history, params)

        print("ChatGPT: ", reply)
        if params.debug:
            print("tokens_num: ", tokens_num)
            print("finish_reason: ", finish_reason)
            print("chat_history: ", chat_history)

        print("------------------------------------------------------------------------------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # We generally recommend altering top_p or temperature but not both.
    parser.add_argument('--temperature',            action='store',        type=float,           default=1,                                       help='Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic(0-2).')
    parser.add_argument('--top_p',                  action='store',        type=float,           default=1,                                       help='0.1 means only the tokens comprising the top 10% probability mass are considered(0-1).')   
    parser.add_argument('--n',                      action='store',        type=int,             default=1,                                       help='How many completions to generate for each prompt.')
    parser.add_argument('--stop',                   action='store',        type=int,             default=None,                                    help='Up to 4 sequences where the API will stop generating further tokens.')
    parser.add_argument('--max_tokens',             action='store',        type=int,             default=20,                                      help='Max token of the answer(0-4096)')
    parser.add_argument('--presence_penalty',       action='store',        type=float,           default=0,                                       help='Positive values penalize new tokens based on whether they appear in the text so far, increasing the model likelihood to talk about new topics(-2-+2).')
    parser.add_argument('--frequency_penalty',      action='store',        type=float,           default=0,                                       help='Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the models likelihood to repeat the same line verbatim(-2-+2).')   
    
    parser.add_argument('--use_system',             action='store',        type=bool,            default=True,                                    help='Whehter to input system message at the beginning.') 
    parser.add_argument('--system_message',         action='store',        type=str,             default="Your answer must be within 20 words",   help='What to input system message at the beginning.')  

    parser.add_argument('--debug',                  action='store',        type=bool,            default=False,                                   help='Whehter print the debug information(tokens_num, finish_reason, chat_history).')                
   
    params = parser.parse_args()
    
    chat(params)
