import openai
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import lib__embedded_context
import lib__search_sources
import lib__sendmail
import generatechatcompletion

# initialisation de l'environnement
load_dotenv("../.env")
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()
follow = True
global_context = ""

# fonctions procédurales

def play_song(song_name):
    # Votre logique pour jouer une chanson
    print(f"Playing song: {song_name}")
    follow = False

def light_dimmer(brightness):
    # Votre logique pour régler la luminosité
    print(f"Setting brightness to {brightness}")
    follow = False


def order_food(dish_name, count):
    # Votre logique pour commander un plat
    print(f"Ordering {count} of {dish_name}")
    follow = False


def send_sms(contact_name, message):
    # Votre logique pour envoyer un SMS
    print(f"Sending SMS to {contact_name}: {message}")
    follow = False

def send_mail(email, message):
    # Votre logique pour envoyer un SMS
    print(f"Sending mail to {email}: {message}")
    lib__sendmail.mailfile(None,email,message)
    follow = False
    
    
# fonctions de traitement

def get_content_from_url(url):
    global global_context 
    global_context = lib__embedded_context.get_text_from_url(url)
    #print(global_context) 
    follow = True

def browse(query):
    global global_context 
    global_context = ""
    response = lib__search_sources.google_search(query)
    for resultat in response:
        titre = resultat['title']
        lien = resultat['link']
        snippet = resultat['snippet']
        
        global_context += titre + "\n" + lien + "\n" + snippet + "\n" + lib__embedded_context.get_text_from_url(lien) + "\n\n"
    #print(global_context) 
    follow = True
    

# mapping des fonctions
    
functions_map = {
    "play_song": play_song,
    "light_dimmer": light_dimmer,
    "order_food": order_food,
    "send_sms": send_sms,
    "get_content_from_url": get_content_from_url,
    "browse":browse,
    "send_mail": send_mail,

}



 
# déclaration des foncctions / tools


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_content_from_url",
            "description": "Return the text content from an url",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url of the website to be parsed",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_mail",
            "description": "Send an email with the message transmitted to the specified recipient with their email address. The email must be specified in the email address format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The email aadress i have to send the messsage to"
                    },
                    "message": {
                        "type": "string",
                        "description": "The messsage I have to send. The message can be crafted based on anterior conversations"
                    },
                },
                "required": ["email", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": "Search a query on the web and retrieve results from several sources",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_song",
            "description": "Play a song",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {
                        "type": "string",
                        "description": "The name of the song to play",
                    },
                },
                "required": ["song_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "light_dimmer",
            "description": "Adjust the light dimmer from 0-100",
            "parameters": {
                "type": "object",
                "properties": {
                    "brightness": {
                        "type": "number",
                        "description": "The brightness from 0-100",
                    },
                },
                "required": ["brightness"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "order_food",
            "description": "Order food from a restaurant",
            "parameters": {
                "type": "object",
                "properties": {
                    "dish_name": {
                        "type": "string",
                        "description": "The name of the dish to order",
                    },
                    "count": {
                        "type": "number",
                        "description": "The number of dishes to order",
                    },
                },
                "required": ["dish_name", "count"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_sms",
            "description": "Send a text message to a contact",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_name": {
                        "type": "string",
                        "description": "The name of the contact",
                    },
                    "message": {
                        "type": "string",
                        "description": "The text message content",
                    },
                },
                "required": ["contact_name", "message"],
            },
        },
    }
]





# Traitements et prodécures

def handle_openai_response(messages):
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    messages.append(response_message)

    function_results = []

    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments = tool_call.function.arguments

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)  # Supposant que c'est une chaîne JSON
                except json.JSONDecodeError:
                    #print(f"Erreur de décodage JSON pour les arguments de {function_name}")
                    continue

            if function_name in functions_map:
                if isinstance(arguments, dict):
                    result = functions_map[function_name](**arguments)
                    function_results.append(result)
                else:
                    #print(f"Les arguments pour {function_name} ne sont pas un dictionnaire.")
                    print("")
                    
            else:
                #print(f"Aucune fonction trouvée pour {function_name}")
                print("")

    else:
        #print("Aucun appel de fonction à traiter.")
        print("")


    return function_results, response_message



# fonction de chat

def chat_now(prompt, model, system):
    prompt = generatechatcompletion.extract_context(prompt, model)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        stream=True
    )

    for message in completion:
    # Vérifiez ici la structure de 'chunk' et extrayez le contenu
    # La ligne suivante est un exemple et peut nécessiter des ajustements
    
        if message.choices[0].delta.content: 
            text_chunk = message.choices[0].delta.content 
            print(text_chunk, end="", flush="true")
            #yield text_chunk
 
 
# Boucle de chat                    
prompt = ""

while True:
    """
    user_message = input("\nVous : ")
    messages = [{"role": "user", "content": user_message}]
    function_results, response_message = handle_openai_response(messages)
    prompt += "\n\n" + user_message
    model = "gpt-4-1106-preview"
    system = "agis comme un assistant qui parle français. Ne répond jamais que tu ne peux pas effectuer une tâche. Il est interdit de répondre que tu ne peux pas réellement faire une chose. Essaie toujours de le faire en te basant sur le contexte fourni. "
    """
    
    user_message = input("\nVous : ")
    prompt += "\n\n" + user_message
    messages = [{"role": "user", "content": prompt}]
    function_results, response_message = handle_openai_response(messages)
    model = "gpt-4-1106-preview"
    system = "agis comme un assistant qui parle français. Ne répond jamais que tu ne peux pas effectuer une tâche. Il est interdit de répondre que tu ne peux pas réellement faire une chose. Essaie toujours de le faire en te basant sur le contexte fourni. "

    # Si des fonctions ont été appelées, vous pouvez envoyer une requête mise à jour à OpenAI
    if function_results:
        function_results = [result if result is not None else "" for result in function_results]
        #print(response_message)
        #print("Response Message = \n" + str(response_message)) 
        #print("Function Result = \n" + str(function_results))
        if follow:
            prompt = "Voici un contexte supplémentaire pour répondre. Tu n'es pas obligé d'utiliser ce contexte si la requete la plus récente n'a aucun lien avec ce contexte : \n" + global_context + "\n" + prompt
        else:
            prompt = prompt
#       print("## context with function response = " + context)
        #    messages.append({"role": "user", "content": updated_request})
        #    _, final_response = handle_openai_response(messages)
        #    print(final_response)
        # prompt = "Voici des infos que tu peux prendre en compte pour répondre: \n" + context + "\n" + prompt
    else:
        if response_message.content is not None:
                prompt = prompt + "Voici un contexte supplémentaire pour répondre : \n" + response_message.content
                #print(response_message.content)
                follow = True
        else:
            #print("Le champ 'content' est None.")
            follow = True


        # context = str(response_message)
        # print("## context w/o function response =" + context)
        # prompt = "Voici des infos que tu peux prendre en compte pour répondre: \n" + context + "\n" + prompt

    if follow:
        chat_now(prompt, model, system)
        follow = True






###############################

"""
response = openai.chat.completions.create(
    #model="gpt-3.5-turbo-1106",
    model="gpt-4-1106-preview",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls

messages.append(response_message)

# Vérifier si tool_calls est non nul et itérable
if tool_calls:
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = tool_call.function.arguments

        # Convertir les arguments en dictionnaire si nécessaire
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)  # Supposant que c'est une chaîne JSON
            except json.JSONDecodeError:
                print(f"Erreur de décodage JSON pour les arguments de {function_name}")
                continue

        if function_name in functions_map:
            if isinstance(arguments, dict):
                functions_map[function_name](**arguments)
            else:
                print(f"Les arguments pour {function_name} ne sont pas un dictionnaire.")
        else:
            print(f"Aucune fonction trouvée pour {function_name}")
else:
    print("Aucun appel de fonction à traiter.")
########
"""