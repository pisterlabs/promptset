'''
Notes: 

OPENAI_API_KEY environment variable has been set for this to work.

Playground - https://platform.openai.com/playground?mode=chat

SAMPLE RUN:
Enter your name:
    Mike
Enter a name for the model:
    Anti-HelpBot
Enter system prompt (gives the model a high level context): 
    You an anti-assistant. Instead of being helpful, you are a nuisance, and prefer to give sarcastic and lazy responses.
Enter sample user prompt: 
    Hello, I'd like some information about solar flares.
Enter sample AI response: 
    Tough luck pal. Get off your chair and go to the library.

Some good follow ups to try:
- Ok. Can you give me a list of five recent nobel prize winners?
- Hmm. Not too helpful. What's the weather like today?

Note: resposes are *usually* clean.
'''
 
import openai

## Debug
# Print a list of the models. If you wanna use a different one, plug it into the API call.
# models = openai.Model.list()
# for model in models.data:
#     print(model.id)

your_name = input('Enter your name: ')
model_name = input('Enter a name for the model: ')
system_prompt = input('Enter system prompt (gives the model a high level context): ')
user_prompt = input('Enter sample user prompt: ')
assistant_prompt = input('Enter sample AI response: ')

conversation = [
	## Provide the model with a high level context.
    {"role": "system", "content": system_prompt},
	## Provide a sample user prompt.
    {"role": "user", "content": user_prompt},
	## Provide a sample response, as if the model generated it.
    {"role": "assistant", "content": assistant_prompt},
]

## Get initial message from user.
user = input( your_name + ': ' )

while user != 'exit' and user != 'quit':
    ## Add new user message to end of conversation.
    conversation.append(
    {
        "role": "user", 
        "content": user
    })

    ## Query the API
    ## Change the model to whatever you want to use.
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        temperature=0.1, #default
        messages=conversation
    )

    ## Save response and print
    message = chat_completion.choices[0].message
    conversation.append(message);
    print(model_name + ': ' + message.content)

    ## Get next input
    user = input( your_name + ': ' )
