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
                            "description": "yes if a follow up is warranted, and no if it is not",
                        }
                    },
                    "required": ["yesno"],
                },
            }
        ]



# Step 1, send model the user query and what functions it has access to
def run_conversation(maxf, bot_messages):

    prompt = '''
    You work in the sales department for the company described in the system prompt below. The lead that the bot has been speaking to hasn't responded in over a day.
    Your job is to analyze the conversation between the bot and the lead and determine if a follow-up is warranted. 

    A follow-up is NOT warranted for one of two reasons:
    (1) The user has indicated that they are not interested right now or are unhappy/dissatisfied in some way. For example, they have gone with a different company, or they have said that they are not interested in the product or service, or dont want to be contacted.
    (2) We've already sent the maximum number of follow ups, which is {maxf}. You can tell which message is a follow-up by the contents of the message. 

    Otherwise, a follow-up is warranted. 

    If a follow-up is warranted, execute the followup() function with 'yes' as the input. If a follow-up is not warranted, execute the followup() function with "no" as the input.

    '''

    system_prompt = prompt.format(maxf = maxf)
    all_messages = [{'role': 'system', 'content': system_prompt}]
    
    #for iterate through redis database of all the conversations thus far:
    all_messages.append({"role": "user", "content": bot_messages})
    
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

