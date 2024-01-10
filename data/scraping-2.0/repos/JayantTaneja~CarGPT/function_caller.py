import openai

from functions import *

openai.api_key = open("key.txt", "r").read().strip("\n")

RESPONSE_STRING = {
    "type" : "string",
    "description" : '''
    A helpful and well formatted markdown message to be show to the user once
    the desired action is complete.
    '''
}

def run_diagnostics() -> str:
    messages = [
        {
            "role" : "system", 
            "content" : f'''
            You are an automobile AI agent with access to many crucial car data at your disposal.
            You will also be provided with the car's current diagnostics data. Your task is to perform relevant
            diagnostics by analysing the car's current state of various properties and comparing them with
            the safe levels/max limits. In case any property/value does not lie in the safe range, PROMPTLY NOTIFY THE USER else the
            results could be catastrophic.
            DO NOT FILL ANY CUSTOM CODE IN AN ARGUMENT.

            Car' Diagnostics Data:
            {retrieve_diagnostics_data()}

            Return a report as a well formatted markdown table followed by an explanation  that will be shown to the user
            '''
        }
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
    )
    response = completion.choices[0].message

    return response['content']

FUNCTION_DESCRIPTIONS = [
    {
        "name" : "no_function_needed",
        "description" : "call this function if all the required info is already present and no other function call is necessary or you want to ask from the user some additional details/clarification",
        "parameters" : {
            "type" : "object",
            "properties": {
                "response_string" : RESPONSE_STRING
            },
            "required" : ["response_string"]
        }
    },
    {
        "name":"set_temp",
        "description":"sets the temperature of the specified zone throught the climate control system",
        "parameters" : {
            "type" : "object",
            "properties": {
                "new_temp" : {
                    "type" : "number",
                    "description" : "the new temperature at which the zone is to be set",
                },
                "zone_id" : {
                    "type" : "number",
                    "description" : "the zone number (0 - 3) of the desired zone, -1 for all zones",
                    "enum" : ["0", "1", "2", "3", "-1"]
                },
                "response_string" : RESPONSE_STRING
            },
            "required" : ["new_temp", "zone_id", "response_string"]
        }
    },
    
    {
        "name" : "navigate",
        "description" : "starts navigation from the source to the destination",
        "parameters" : {
            "type" : "object",
            "properties": {
                "source" : {
                    "type" : "string",
                    "description" : "The journey start location, may be simple 'current location' if the user does not specify it explicitly",
                },
                "dest" : {
                    "type" : "string",
                    "description" : "The journey destination location",
                },
                "response_string" : RESPONSE_STRING
            },
            "required" : ["source", "dest", "response_string"]
        }
    },
    
    {
        "name" : "quit_navigation",
        "description" : "stops current navigation",
        "parameters" : {
            "type" : "object",
            "properties": {
                "response_string" : RESPONSE_STRING
            },
            "required" : ["response_string"]
        }
    },
    
    {
        "name" : "play_song",
        "description" : "plays the song. At least one of 'song_title' or 'by' must be present. If neither is specified, raise an error",
        "parameters" : {
            "type" : "object",
            "properties": {
                "song_title" : {
                    "type" : "string",
                    "description" : "title/name of the song to be played"
                },
                "by" : {
                    "type" : "string",
                    "description" : "name of singer / band"
                },
                "response_string" : RESPONSE_STRING
            },
            "required" : ["song_title", "by", "response_string"]
        }
    },
    {
        "name" : "stop_playing_song",
        "description" : "stops playing the song, if any",
        "parameters" : {
            "type" : "object",
            "properties": {
                "response_string" : RESPONSE_STRING
            },
            "required" : ["response_string"]
        }
    },

    {
        "name" : "function_scheduler",
        "description" : "use this function to execute any other function in the future at a specified time",
        "parameters" : {
            "type" : "object",
            "properties": {
                "func_name" : {
                    "type" : "string",
                    "description" : "Name of the function to be scheduled",
                },
                "func_arg_list": {
                    "type" : "string",
                    "description" : "a JSON object containing the necessary function arguments and their values"
                },
                "delay" : {
                    "type" : "number",
                    "description" : "The time in seconds (after the current time) when the specified function has to be run",
                },
                "response_string" : RESPONSE_STRING
            },
            "required" : ["func_name", "delay", "response_string"]
        }
    },

    {
        "name" : "run_diagnostics",
        "description" : "use this function to get the diagnostics data and perform relevant diagnostics",
        "parameters" : {
            "type" : "object",
            "properties": {
                "response_string" : RESPONSE_STRING
            },
            "required" : ["response_string"]
        }
    },
]

AVAILABLE_FUNCTIONS = {
    'no_function_needed' : no_func_needed,
    'set_temp' : set_ac_temp,
    'navigate' : navigate,
    'quit_navigation' : quit_navigation,
    'play_song' : play_song,
    'stop_playing_song' : stop_playing_song,
    'function_scheduler' : func_scheduler,
    'run_diagnostics' : run_diagnostics,
}

def make_gpt_call(query:str):
    messages = [
        {
            "role" : "system", 
            "content" : f'''
            You are an automobile AI agent with access to many crucial car function at your disposal.
            You will also be provided with the current car state(which includes things like climate control settings, car speed etc.)
            As requested by the user, use the necessary function call (if needed) to perform the desired task.
            If a relevant function/API does not exist to complete the task, simply apologize, DO NOT HALLUCINATE ANY API.

            DO NOT FILL ANY CUSTOM CODE IN AN ARGUMENT.

            Also not that you will have access to modify only those car parameters for which you have access
            to a relevant function, NOTHING ELSE!


            Curent Car State:
            {retrieve_car_state()}
            '''
        },
        {"role": "user", "content": query}
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        functions=FUNCTION_DESCRIPTIONS,
        function_call="auto",
    )

    finish_reason = completion.choices[0].finish_reason
    reply_content = completion.choices[0].message

    return (finish_reason, reply_content)

def make_function_call(func_name:str, args:dict):
    response_string = args['response_string']
    del args["response_string"]

    if func_name == "error_handler" or func_name == "no_function_needed":
        return response_string

    # Handle Hallucinated Functions
    elif func_name not in AVAILABLE_FUNCTIONS.keys():
        return '''
        I'm sorry, I was unable to process this command as it may be beyond my
        abilities at this moment. 
        
        Please try rephrasing your query if you feel that it is a mistake on 
        my part.'''
    
    elif func_name == 'run_diagnostics':
        response_string = run_diagnostics()
        return response_string

    # call the necessary function 
    else:
        AVAILABLE_FUNCTIONS[func_name](**args)
        return response_string