import openai

openai.api_key = "sk-<<Your API key here>>"
messages = []


#Append the message to the conversation history 
def add_message(role, message):
    messages.append({"role": role, "content": message})

def converse_with_chatGPT():
    model_engine = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=model_engine, #Open AI model name
        messages=messages, # user query
        max_tokens = 1024, # this is the maximum number of tokens that can be used to provide a response.
        n=1, #number of responses expected from the Chat GPT
        stop=None, 
        temperature=0.5 #making responses deterministic
    )
    # print(response)
    message = response.choices[0].message.content
    return message.strip()

# process user prompt
def process_user_query(prompt):
    user_prompt = (f"{prompt}")
    add_message("user", user_prompt)
    result = converse_with_chatGPT()
    print(result)

#Request user to provide the query
def user_query():
    while True:
        prompt = input("Enter your question: ")
        response = process_user_query(prompt)
        print(response)

user_query()