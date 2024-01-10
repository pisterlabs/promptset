
from functions import find_examples
import os
from datetime import datetime
from supabase import create_client, Client
import openai

#connect to supabase database
urL: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(urL, key)

now = datetime.now()
now = now.strftime("%Y-%m-%d %H:%M:%S")


def generate_response(input_message):

    #variables for system prompt
    info = {
        'bot_name': 'Taylor',
        'membership_link': 'https://www.reposite.io/membership-overview-1',
        'email': 'john@doe.com',
        'supplier_name': 'Acme Trading Co',
        'lead_first_name': 'John',
        'lead_last_name': 'Doe',
        'nmqr_count': '10',
        'nmqrurl': 'nmqrurl.com',
        'reseller_org_name': 'Smith Co',
        'category': 'travel',
        'date': 'June 20, 2023',
        'current_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'destination': 'Honolulu',
        'group_size': '50',
        'trip_dates': 'August 10, 2023 to August 20, 2023'
    }

    initial_text  = '''
        Hey {lead_first_name} -

I saw that your Reposite profile just sparked some new interest! A planner from {reseller_org_name} just sent you a new quote request - they're looking for {category} suppliers in {destination}.

Based on the details, do you feel like this lead is relevant for {supplier_name}?

Cheers,
Taylor
'''
    initial_text = initial_text.format(**info)
    

    data, count = supabase.table("bots_dev").select("*").eq("id", "taylor").execute()      
    bot_info = data[1][0]
    system_prompt = bot_info['system_prompt']


    #extract examples 
    examples = find_examples(input_message, k = 6)
    print(examples)
    print('\n\n')
    system_prompt = system_prompt + '\n\n' + examples
    system_prompt = system_prompt.format(**info)
    system_prompt = {'role': 'system', 'content': system_prompt}

    initial_text = {'role': 'assistant', 'content': initial_text}
    user_response = {"role": "user", "content": input_message}

    messages = []
    messages.append(system_prompt)
    messages.append(initial_text)
    messages.append(user_response)

    response = openai.ChatCompletion.create(
        messages = messages,
        model = 'gpt-4',
        max_tokens = 400
    )

    response = response["choices"][0]["message"]["content"]
    return response


