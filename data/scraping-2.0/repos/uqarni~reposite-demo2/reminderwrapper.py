import openai
import json


def followup(yesno):
    if yesno == 'yes':
        return 'y'
    if yesno == 'no':
        return 'n'
    else:
        return 'n'



########
functions=[
            {
                "name": "followup",
                "description": "Analyze the conversation provided by the user and determine if a followup is warranted",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "yesno": {
                            "type": "string",
                            "enum": ["yes", "no"],
                            "description": "yes if a follow up is warranted, and no if it is not",
                        }
                    },
                    "required": ["yesno"],
                },
            }
        ]



# Step 1, send model the user query and what functions it has access to
def run_conversation(bot_messages):

    prompt = '''
    You work in the sales department for Reposite, a travel agency and experience supplier marketplace.
    Your job is to analyze the conversation between our sales agent (the Assistant) and the potential customer (the User) and determine if a follow-up is warranted. 

    A follow-up is NOT warranted if:
    (1) the user has indicated that they are not interested or are unhappy in some way. For example, they have said that they are not interested in the product or do not want to be contacted.
    (2) the user has indicated that they already purchased the Reposite membership.
    

    Otherwise, a follow-up is warranted. 

    If a follow-up is warranted, execute the followup() function with 'yes' as the input. If a follow-up is not warranted, execute the followup() function with "no" as the input.
    '''
    all_messages = [{'role': 'system', 'content': prompt}]
    
    #for iterate through redis database of all the conversations thus far:
    all_messages.append({"role": "user", "content": str(bot_messages)})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=all_messages,
        functions=functions,
        function_call= {"name": "followup"}

    )

    message = response["choices"][0]["message"]

    # Step 2, check if the model wants to call a function
    if message.get("function_call"):
        function_name = message["function_call"]["name"]
        return json.loads(message["function_call"]["arguments"])["yesno"]
    else:
        return 'error'

