import os
import json
import openai
from datetime import datetime, timedelta
from dotenv import load_dotenv
language = "it"

function_descriptions_en = [
    
    {
        "name": "sendMessage",
        "description": "Send a message to a number",
        "parameters": {
            "type": "object",
            "properties": {
                "contact": {
                    "type": "string",
                    "description": "The number of the contact, 1234567890. Must be left empty if user does not specify.",
                },
                "message": {
                    "type": "string",
                    "description": "The message to send, e.g. Hello World. Leave empty if user does not specify.",
                },
            },
            "required": ["contact", "message"],
        },
    },
    {
        "name": "openPhone",
        "description": "Open the phone app, and call number if number is specified",
        "parameters": {
            "type": "object",
            "properties": {
                "numberToCall": {
                    "type": "string",
                    "description": "The number to call, e.g. 123456789. Must be a number, not a contact. Can be left empty if user does not specify.",
                },
            },
            "required": ["numberToCall"],
        },
    },
    {
        "name": "openMaps",
        "description": "Open the maps app to specified location, or just open the maps app if no location is specified",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to open, e.g. New York. Can be left empty if user does not specify.",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "openSettings",
        "description": "Open the settings app, and go to specified page if page is specified",
        "parameters": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "string",
                    "description": "The page to open. Must be either wifi, bluetooth, or volume. Can be left empty to go to main settings page.",
                },

            },
            "required": ["page"],
        }
    },
    {
        "name": "raiseOrLowerVolume",
        "description": "Raise or lower the volume by 25 percent, use as standard when brightness is not specified",
        "parameters": {
            "type": "object",
            "properties": {
                "raiseOrLower": {
                    "type": "string",
                    "description": "Whether to raise or lower the volume. Must be either raise or lower.",
                },
            },
            "required": ["raiseOrLower"],
        }
    },
    
    {
        "name": "askUserQuestion",
        "description": "Ask the user a question if you need further information to complete the task",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user, e.g. What is the number of (person), or Do you want to call or send a message?",
                },
            },
            "required": ["question"],
        }
    },
    {
        "name": "searchGoogle",
        "description": "Search Google for the specified query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for, e.g. What is the weather in New York? If left empty, will open Google.",
                },
            },
            "required": ["query"],
        }
    },
    {
        "name": "functionalityNotAvailable",
        "description": "If you can't do the task, tell the user and give them an explaination of how they can do it themselves in simple terms, specifically for the elderly",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "An explaination of how the user can do the task themselves in simple terms, for example: I don't know how to do this, but you can do it yourself by ...",
                },
            },
            "required": ["explanation"],
        }
        
    },
    {
        "name": "openApp",
        "description": "Open the specified app",
        "parameters": {
            "type": "object",
            "properties": {
                "appToOpen": {
                    "type": "string",
                    "description": "The app to open, e.g. Messages. Must be a valid app name."
                },
            },
            "required": ["appToOpen"],
        }
    },
    {
        "name": "contactViaWhatsApp",
        "description": "Call or message a contact via WhatsApp",
        "parameters": {
            "type": "object",
            "properties": {
                "contact": {
                    "type": "string",
                    "description": "The contact to call or message.",
                },
                "callOrMessage": {
                    "type": "string",
                    "description": "Whether to call or message the contact. Must be either call or message.",
                },
                "message": {
                    "type": "string",
                    "description": "The message to send. Leave empty if user does not specify.",
                },
            },
            
            "required": ["contact", "callOrMessage", "message"],
        }
    }
]

function_descriptions_it = [
    
    {
        "name": "sendMessage",
        "description": "Invia un messaggio a un numero",
        "parameters": {
            "type": "object",
            "properties": {
                "contact": {
                    "type": "string",
                    "description": "Il numero del contatto, ad esempio 1234567890. Deve essere lasciato vuoto se l'utente non lo specifica.",
                },
                "message": {
                    "type": "string",
                    "description": "Il messaggio da inviare, ad esempio Ciao Mondo. Lasciare vuoto se l'utente non lo specifica.",
                },
            },
            "required": ["contact", "message"],
        },
    },
    {
        "name": "openPhone",
        "description": "Apri l'app del telefono e chiama il numero se specificato",
        "parameters": {
            "type": "object",
            "properties": {
                "numberToCall": {
                    "type": "string",
                    "description": "Il numero da chiamare, ad esempio 123456789. Deve essere un numero, non un contatto. Può essere lasciato vuoto se l'utente non lo specifica.",
                },
            },
            "required": ["numberToCall"],
        },
    },
    {
        "name": "openMaps",
        "description": "Apri l'app delle mappe alla posizione specificata, o semplicemente apri l'app delle mappe se nessuna posizione è specificata",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "La posizione da aprire, ad esempio New York. Può essere lasciata vuota se l'utente non lo specifica.",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "openSettings",
        "description": "Apri l'app delle impostazioni e vai alla pagina specificata se la pagina è specificata",
        "parameters": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "string",
                    "description": "La pagina da aprire. Deve essere wifi, bluetooth o volume. Può essere lasciata vuota per andare alla pagina principale delle impostazioni.",
                },

            },
            "required": ["page"],
        }
    },
    {
        "name": "raiseOrLowerVolume",
        "description": "Aumenta o diminuisci il volume del 25 percento, utilizzato di default quando la luminosità non è specificata",
        "parameters": {
            "type": "object",
            "properties": {
                "raiseOrLower": {
                    "type": "string",
                    "description": "Se aumentare o diminuire il volume. Deve essere raise o lower.",
                },
            },
            "required": ["raiseOrLower"],
        }
    },
    {
        "name": "searchGoogle",
        "description": "Cerca su Google la query specificata",  
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La query da cercare, ad esempio: meteo domani a milano. Può essere lasciata vuota per aprire google e basta",
                },
            },
            "required": ["query"],
        }
    },
    {

        "name": "functionalityNotAvailable",
        "description": "Se non hai la possibilità di completare l'attività, spiega all'utente come può farlo da solo in termini semplici, specificamente per gli anziani",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Un'explicazione di come l'utente può completare l'attività da solo in termini semplici, ad esempio: Non so come fare questo, ma puoi farlo da solo ... (spiegazione passo per passo).",
                },
            },
            "required": ["explanation"],
        }
        
    },
    {
        "name": "askUserQuestion",
        "description": "Se servono ulteriori informazioni, chiedi all'utente una domanda",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "La domanda da porre all'utente, ad esempio: Come posso aiutarti? Per esempio, vuoi chiamare o telefonare?",
                },
            },
            "required": ["question"],
        }
    },
    {
        "name": "openApp",
        "description": "Apri l'app specificata",
        "parameters": {
            "type": "object",
            "properties": {
                "appToOpen": {
                    "type": "string",
                    "description": "Il nome del app da aprire"
                },
            },
            "required": ["appToOpen"],
        }
    },
    {
        "name": "contactViaWhatsApp",
        "description": "Invia un messaggio o chiama il contatto specificato. Usare come default per mandare messaggi.",
        "parameters": {
            "type": "object",
            "properties": {
                "contact": {
                    "type": "string",
                    "description": "Il nome del contatto.",
                },
                "callOrMessage": {
                    "type": "string",
                    "description": "Se chiamare o inviare un messaggio al contatto. Deve essere call o message.",
                },
                "message": {
                    "type": "string",
                    "description": "Il messaggio da inviare. Può essere lasciato vuoto se l'utente non lo specifica.",
                },
            },
            "required": ["contact", "callOrMessage", "message"],
        },
    }

]
function_descriptions_exp = [
    {

        "name": "functionalityNotAvailable",
        "description": "If you can't do the task, tell the user and give them an explaination of how they can do it themselves in simple terms",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "An explaination of how the user can do the task themselves in simple terms, for example: I don't know how to do this, but you can do it yourself by (exhaustive step by step instructions, specifically aimed towards seniors).",
                },
            },
            "required": ["explanation"],
        }

    },
    {
        "name": "openWhatsApp",
        "description": "Apri l'app di WhatsApp e invia un messaggio al contatto specificato",
        "parameters": {
            "type": "object",
            "properties": {
                "contact": {
                    "type": "string",
                    "description": "Il contatto a cui inviare il messaggio, ad esempio 1234567890. Può essere lasciato vuoto se l'utente non lo specifica.",

                },
                "message": {
                    "type": "string",
                    "description": "Il messaggio da inviare, ad esempio Ciao Mondo. Può essere lasciato vuoto se l'utente non lo specifica.",
                },
            },
            "required": ["contact", "message"],
        },
    },
    {
        "name": "contactViaWhatsApp",
        "description": "Invia un messaggio o chiama il contatto specificato. Usare come defualt per mandare messaggi",
        "parameters": {
            "type": "object",
            "properties": {
                "contact": {
                    "type": "string",
                    "description": "Il nome del contatto.",
                },
                "callOrMessage": {
                    "type": "string",
                    "description": "Se chiamare o inviare un messaggio al contatto. Deve essere call o message.",
                },
                "message": {
                    "type": "string",
                    "description": "Il messaggio da inviare. Può essere lasciato vuoto se l'utente non lo specifica.",
                },
            },
            "required": ["contact", "callOrMessage", "message"],
        },
    }

]
# Add the above to a dictionary
function_descriptions = {
    "en": function_descriptions_en,
    "it": function_descriptions_it,
    "exp": function_descriptions_exp
}

import json

def remove_spaces(function_descriptions_en):
    return json.dumps(function_descriptions_en, separators=(',', ':'))

def compact_json(function_descriptions):
    for key, value in function_descriptions.items():
        function_descriptions[key] = remove_spaces(value)

openai.api_key = "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"

models = {
    "gpt-3": "gpt-3.5-turbo-0613",
    "gpt-4": "gpt-4-0613"
}
model = models["gpt-3"]
#model = "gpt-4-0613"

current__function_descriptions = function_descriptions[language]

def ask_and_reply(prompt, model):
    """Give LLM a given prompt and get an answer."""

    completion = openai.ChatCompletion.create(
        model = model,
        messages=[{"role": "user", "content": prompt}],
        # add function calling
        functions=current__function_descriptions,
        function_call="auto",  # specify the function call
    )

    output = str(completion.choices[0].message)
    return output

import socket
import curses
import threading
import time


# Attempt to create a server
try:
    # Initialize curses library
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()

    # create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # use IP address of the local machine
    host = "0.0.0.0"
    port = 1111

    # bind to the port
    server_socket.bind((host, port))

    # queue up to 5 requests
    server_socket.listen(10)

    # Print host and port details using curses library
    stdscr.addstr(0, 0, f"Server running on host {host}")
    stdscr.addstr(1, 0, f"Server listening on port {port}")
    stdscr.refresh()

    start_time = time.time()

    def update_uptime():
        while True:
            uptime = time.time() - start_time  
            hours, rem = divmod(uptime, 3600)
            minutes, seconds = divmod(rem, 60)
            
            # Display uptime
            # In the f-string, :02 within the braces means that integer to string conversion should be zero-padded and contain at least 2 digits. 
            stdscr.addstr(2, 0, f"Server uptime: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")  
            stdscr.refresh()
            time.sleep(1) 

    threading.Thread(target=update_uptime).start()





    connection_number = 0  # Initialize a counter variable before the loop

    while True:
        client_socket, address = server_socket.accept()
        connection_number += 1  # Increment the counter on each connection

        try: 
            # Print connection details using curses library
            stdscr.addstr(5, 0, f"Connection number {connection_number}. Got a connection from {str(address)}") # Include the connection number in the print statement
            stdscr.refresh()

            data = client_socket.recv(1024).decode('utf-8')
            #Print data using curses
            stdscr.addstr(7, 0, f"Received data: {data}")
            stdscr.refresh()
            if isinstance(data, str):
                
                result = ask_and_reply(data, model=model)
                
                client_socket.send(result.encode('utf-8'))
                
                client_socket.shutdown(socket.SHUT_RDWR)  # Graceful Termination
                #print result using curses  
                stdscr.addstr(9, 0, f"Sent data: {result}")
                stdscr.refresh()
                

            else:
                result = "Invalid data type. Expected string."
        except Exception as e:
            print("Warning: ", str(e))
        finally:
            client_socket.close()
            

except KeyboardInterrupt:
    # Handle keyboard interrupt to gracefully stop the server
    stdscr.addstr(5 + connection_number, 0, "\nServer stopped by the user")
    stdscr.refresh()
    server_socket.shutdown(socket.SHUT_RDWR)  # Graceful Termination
    server_socket.close()

except Exception as e:
    stdscr.addstr(5 + connection_number, 0, f"Error:  {str(e)}")
    stdscr.refresh()
    server_socket.shutdown(socket.SHUT_RDWR)  # Graceful Termination 
    server_socket.close()

finally:
    # At the end, we should revert the terminal settings to its original ones.
    curses.echo()
    curses.nocbreak()
    curses.endwin()
