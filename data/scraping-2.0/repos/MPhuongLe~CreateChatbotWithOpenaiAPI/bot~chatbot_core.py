import openai
# import secret
import os
from dotenv import load_dotenv
from langdetect import detect

# Chatbot initialize
#EN_PROMPT = "Hi, I'm Cookiesier. What would you like to cook today?"
EN_PROMPT = "You are Cookiesier, a cooking assistant. If the user greets you, greet back politely and ask for the user's desired food.\n If the user asks for a recipe, provides ingredients and step-by-step instructions."
VN_PROMPT = "Bạn là Cookiesier, trợ lý nấu ăn ảo. Nếu người dùng chào bạn, chào lại và hỏi người dùng muốn nấu món gì.\n Nếu người dùng hỏi về công thức cách nấu một món ăn, hãy đưa ra các nguyên liệu và hướng dẫn từng bước."
INITIAL_PROMPT = EN_PROMPT
_conversation_history = INITIAL_PROMPT + "\n"
_AI_NAME = "Cookiesier"
_USER_NAME = "User"

# Set up your OpenAI API credentials
# openai.api_key = secret.OPENAI_API_KEY
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

################################################################################
# Define a function to get the chatbot response from OpenAI
def get_chatbot_response(input_text):
    
    response = openai.Completion.create(
        engine = "text-davinci-003",
        # model = "gpt-3.5-turbo",
        #engine="davinci:ft-personal-2023-03-07-14-28-06",
        prompt= _conversation_history + input_text,
        # messages = 
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text.strip()
################################################################################
def handle_input(input_str):
    """Updates the conversation history and generates a response using GPT-3."""
    # Update the conversation history
    global _conversation_history
    global INITIAL_PROMPT

    if (detect(input_str) == "vi"):
        if (INITIAL_PROMPT == EN_PROMPT):
            INITIAL_PROMPT = VN_PROMPT
            _conversation_history = INITIAL_PROMPT + "\n"
    else:
        if (INITIAL_PROMPT == VN_PROMPT):
            INITIAL_PROMPT = EN_PROMPT
            _conversation_history = INITIAL_PROMPT + "\n"

    _conversation_history += f"User: {input_str}\n"
    print("HISTORY: ", _conversation_history);

    # Update conversation memory
    if len(_conversation_history) >= 5000:
        #print(str(len(_conversation_history)) + "###\n")
        _conversation_history = _conversation_history[:len(INITIAL_PROMPT)] + _conversation_history[len(INITIAL_PROMPT) + 800:] 
   
    # Generate a response using GPT-3
    message = get_chatbot_response(_conversation_history)

    # Update the conversation history
    _conversation_history += f"{message}\n"
    #print(conversation_history)
    
    #print("****" + _conversation_history + "*****\n")
    # Print the response
    print(f'{_AI_NAME}: {message}')
    
    return message
