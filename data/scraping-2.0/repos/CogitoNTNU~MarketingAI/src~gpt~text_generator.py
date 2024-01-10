import openai
from src.config import Config


try:
    # Set OpenAI API key
    api_key = Config().API_KEY
    openai.api_key = api_key
except:
    print("OpenAI API key not found. Please set the environment variable OPENAI_API_KEY to your API key.")
    exit(1)


def request_chat_completion(previous_message: dict, role: str = "system", message: str = "", functions: list = []): 
   
    # previous_message = get_system_prompt()

    try:
        if(not (role == "system" or "user" or "assistant")):
            print("Invalid role")
            return ""
        

        if(previous_message):
            response = openai.ChatCompletion.create(
                model = Config().GPT_MODEL,
                messages = [
                    previous_message,
                    {"role": role, "content": message}
                ], 
                functions = functions
            )
        else: 
            response = openai.ChatCompletion.create(
                model = Config().GPT_MODEL,
                messages=[
                    {"role": role, "content": message}, 
                ]
            )
        return response["choices"][0]["message"]["content"]
    
    except Exception as error: 
        print(f"An error has occured while requesting chat completion.")
        print(f"The error: {str(error)}")
        return ""