import os
import openai
import sys
import argparse
import requests
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import io
from PIL import Image
#import win32clipboard

def setup_openai(serper_key = False):
    # Setup OpenAI API
    local_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_key = os.path.join(local_dir, 'openai_key.txt')

    try:
        with open(path_to_key, 'r') as file:
            openai_key = file.read()
    except Exception as e:
            print(e)

    openai.api_key = openai_key
    os.environ['OPENAI_API_KEY'] = openai_key

    if serper_key:
        path_to_key = os.path.join(local_dir, 'serper_key.txt')
        try:
            with open(path_to_key, 'r') as file:
                serper_key = file.read()
        except Exception as e:
            print(e)
        os.environ["SERPER_API_KEY"] = serper_key
        return openai_key, serper_key

    return openai_key

def generate_wheel_elements(prompt, demo=False):

    my_key = setup_openai()

    if demo:
        return "text"

    wheel_keywords = ["text", "code", "web", "image_editor", "stock"]

    messag=[{"role": "system", "content": f"You are a bot what keyboard describes better a certain program, \
            depending on the program name, from this lists of keywords: {wheel_keywords}."}]
    
    # User history to condition the bot - how do we like the answers to be?
    history_user = [f"The answers has to be a single number, 1, 2, 3 or 4 depending on the better definition. Write nothing else"]
    
    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the name of the program used, and ill give the number."]

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messag,
        max_tokens=200,
        temperature=0,
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content
    history_bot.append(result)
    history_user.append(str(prompt))

    if result in ['1', '2', '3', '4']:
        return wheel_keywords[int(result)-1]
    return "Error"

def generate_response_brainstorm(prompt):
    my_key = setup_openai()

    messag=[{"role": "system", "content": "You are a keywords to abstract generator. From the given keywords, you will generate a solid idea for a hackaton. "}]
    
    # User history to condition the bot - how do we like the answers to be?
    history_user = ["i'll give you some key words. with all of them, you will generate a solid idea for a hackaton project. "]
    
    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the keywords."]
    

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messag,
        max_tokens=200,
        temperature=0.7,
    )


    result = ''
    for choice in response.choices:
        result += choice.message.content
    history_bot.append(result)
    history_user.append(str(prompt))
    return result

def generate_response_brainstorm_image(prompt, quality='high'):
    # Generate reference image from prompt
    my_key = setup_openai()

    if quality == 'low':
        size = "256x256"
    elif quality == 'medium':
        size = "512x512"
    elif quality == 'high':
        size = "1024x1024"

    # elif
    # we don't have that much money

    # Generate a nice image prompt from the text/keywords
    messag=[{"role": "system", "content": "You are a keywords to image prompt generator.\
              From the given keywords or message, you will generate a \
             prompt for Dalle to generate a nice reference image. "}]
    history_user = ["i'll give you some key words or a message. with all of them, you will generate a nice reference image. "]
    history_bot = ["Yes, I'm ready! Please provide the keywords or message."]
    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messag,
        max_tokens=200,
        temperature=0.8,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content

    # Generate image from prompt
    response = openai.images.generate(
    prompt=result,
    n=1,
    size=size
    )
    image_url = response.data[0].url

    # Get the image from the url
    response = requests.get(image_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Convert the image content to BMP format
        image = Image.open(io.BytesIO(response.content))

        """
        def send_to_clipboard(image):
            output = io.BytesIO()
            image.convert('RGB').save(output, 'BMP')
            data = output.getvalue()[14:]
            output.close()
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
            win32clipboard.CloseClipboard()

        send_to_clipboard(image)
        """
        # Save image to the local folder
        image.save('reference.png')

    else:
        print(f"Error: {response.status_code} - {response.text}")

    return image


def generate_response_program(prompt):
    my_key = setup_openai()
    messag=[{"role": "system", "content": "You are a code writting bot. From the given text, you will generate code tu fullfill the functions, with no other text or information. "}]
    
    # User history to condition the bot - how do we like the answers to be?
    history_user = ["i'll give you a paragraph with what i want the code to do. write the code with comments and nothing else. "]
    
    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the expected function of the code. "]
    

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messag,
        max_tokens=400,
        temperature=0.7,
    )


    result = ''
    for choice in response.choices:
        result += choice.message.content
    history_bot.append(result)
    history_user.append(str(prompt))
    return result

def generate_response_feedback(prompt):
    my_key = setup_openai()
    messag=[{"role": "system", "content": "You are a feedback bot. From the given text or code, you will generate a feedback. "}]
    
    # User history to condition the bot - how do we like the answers to be?
    history_user = ["i'll give you a paragraph with what i want to give feedback to. write the feedback and nothing else. \
                    take into account wether it is code or text. if its code, you can give feedback to the code itself, \
                    or to the expected output. look out for errors, mistyped things... \
                    if its text, you can give feedback to the ideas, the writing style, or the grammar."]
    
    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the text or code to give feedback to. "]
    

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messag,
        max_tokens=200,
        temperature=0.3,
    )


    result = ''
    for choice in response.choices:
        result += choice.message.content
    history_bot.append(result)
    history_user.append(str(prompt))
    return result

def generate_response_shortcuts(prompt):
    my_key = setup_openai()
    
    messag=[{"role": "system", "content": "You are a shortcuts bot for apple laptops. From the given program and action, \
             you will generate the shortcut for Mac. give only the command "}]

    # User history to condition the bot - how do we like the answers to be?
    history_user = ["i'll give you a program and an action. write the shortcut for it, nothing else. "]

    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the program name and action. ill give nothing more than the answer for the mac "]

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messag,
        max_tokens=200,
        temperature=0,
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content
    history_bot.append(result)
    history_user.append(str(prompt))
    return result

def generate_response_shortcuts_information(prompt):
    my_key = setup_openai()
    
    messag=[{"role": "system", "content": "You are a shortcuts bot for apple laptops. From the given program, \
             you will generate four shortcuts for Mac in a single line, separated by #. the format will be the following: \
             the name as short as possible, ;, an emoji, ;, and the shortcut. \
             include only commands with command or ctrl, dont add line breaks or 'and' "}]
    
    # User history to condition the bot - how do we like the answers to be?         
    history_user = ["i'll give you a program. write four common shortcuts for it with the correct format, nothing else. "]

    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the program name. ill give nothing more than the answer for the mac "]

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messag,
        max_tokens=400,
        temperature=0,
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content
    print(result)
    # Add shortcuts to the dictionary separated by #
    dictionary = {}
    for shortcut in result.split('#'):
        shortcut = shortcut.split(';')
        if len(shortcut) == 3:
            shortcut_name = shortcut[0]
            shortcut_emoji = shortcut[1]
            shortcut_shortcut = shortcut[2]
            dictionary[shortcut_name] = [shortcut_emoji, shortcut_shortcut, "", ""]

    for shortcut in dictionary:
        print(shortcut)

    return dictionary

def generate_response_feedback_image(base64_image):
    my_key = setup_openai()

    path_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'captura.png')

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {my_key}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "You are a feedback bot. From the given image, you will generate a feedback."
            },
            {
            "type": "text",
            "text": "i'll give you an image with what i want to give feedback to. write the feedback and nothing else. take into account the artistic style, the composition, the colors..., if it's a slide for a presentation take readibility into account, and the amount of text. if it's a logo, take into account the colors, the shapes, the font... if it's a graph, take into account the colors, the labels, the axis... and so on"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
 
    assistant_message = response['choices'][0]['message']['content']

    return assistant_message

def generate_response_stocks(prompt, search = True):
    if search:
        my_key, serper_key = setup_openai(serper_key=True)

        llm = OpenAI(temperature=0)
        search = GoogleSerperAPIWrapper()
        tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run,
            description="useful for when you need to ask with search"
            )
        ]

        self_ask_with_search = initialize_agent(tools, llm, agent=AgentType.SELF_ASK_WITH_SEARCH, verbose=True)
        answer = self_ask_with_search.run(prompt)
        return answer

    # Model withouth search
    my_key = setup_openai()
    messag=[{"role": "system", "content": "You are an investing advice bot. given the user query, you will aid them to make a decission or inform them about the stock market. "}]

    # User history to condition the bot - how do we like the answers to be?
    history_user = ["i'll give you a query about the stock market. write the answer. "]

    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the query. ill give nothing more than the answer "]

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messag,
        max_tokens=200,
        temperature=0.7,
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content
    history_bot.append(result)
    history_user.append(str(prompt))
    return result





if __name__ == '__main__':

    my_key = setup_openai()

    # Inputs: 
    # wheel - returns the options for the wheel
    # brainstorm - given some keywords, returns a project idea
    # program - given the description of a program and the language, returns the code
    # feedback - given the selected text, returns a feedback
    # shortcuts - given the program used, returns the common shortcuts

    # Argument parsing 
    parser = argparse.ArgumentParser()
    parser.add_argument('query_type', help='What type of query is this?')
    parser.add_argument('prompt', help='Prompt to be used for the chatbot')
    parser.add_argument('--demo', help='Is the demo mode on?')

    args = parser.parse_args()
    if args.demo == 'True':
        print('Demo mode is on')
        demo = True
    else:
        print('Demo mode is off')
        demo = False
        
    if args.query_type == 'wheel':
        response = generate_wheel_elements(args.prompt, demo)
        print(response)
                
    elif args.query_type == 'brainstorm':
        response = generate_response_brainstorm(args.prompt)
        print(response)

    elif args.query_type == 'brainstorm_image':
        response = generate_response_brainstorm_image(args.prompt)
        print(response)

    elif args.query_type == 'program':
        response = generate_response_program(args.prompt)
        print(response)

    elif args.query_type == 'feedback':
        response = generate_response_feedback(args.prompt)
        print(response)

    elif args.query_type == 'feedback_image':
        response = generate_response_feedback_image(args.prompt)
        print(response)

    elif args.query_type == 'shortcuts':
        response = generate_response_shortcuts(args.prompt)
        print(response)

    elif args.query_type == 'shortcuts_list':
        response = generate_response_shortcuts_list(args.prompt)
        print(response)

    elif args.query_type == 'shortcuts_information':
        response = generate_response_shortcuts_information(args.prompt)
        print(response)

    elif args.query_type == 'usecases':
        response = generate_response_usecases(args.prompt)
        print(response)

    elif args.query_type == 'stocks':
        response = generate_response_stocks(args.prompt, search = False)
        print(response)

