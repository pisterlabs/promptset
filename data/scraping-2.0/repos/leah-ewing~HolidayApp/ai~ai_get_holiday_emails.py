import requests, os, json
import time

ROOT_FOLDER = os.environ['ROOT_FOLDER']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_API_URI = os.environ['OPENAI_API_URI']

api_key = OPENAI_API_KEY
endpoint = OPENAI_API_URI


def ask_question(question):
    """ Asks a question to OpenAI's `/completion` endpoint """

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        'model': 'text-davinci-003',
        'prompt': question,
        'max_tokens': 100
    }

    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['text']
    elif response.status_code == 429:
        return f"Error {response.status_code}: Rate limit hit... Retrying..."
    else:
        return f"Error: {response.status_code}"
    
    
def create_email_json():
    """ Creates a json of holiday names and their email blurbs populated from OpenAI responses """

    new_holiday_emails_json = open(f'{ROOT_FOLDER}/ai/json/new_holiday_emails.json')
    new_json = json.load(new_holiday_emails_json)

    directory = f'{ROOT_FOLDER}/json/holidays'
    for file in os.listdir(directory):
        file_name = os.path.join(directory, file)
        f  = open(file_name)
        holidays = json.load(f)

        for holiday in holidays:
            if holiday['holiday_name'] not in str(new_json):
                question = f"Will you write me a short blurb describing {holiday['holiday_name']}? Keep it around 250-300 characters."
                answer = ask_question(question)

                if answer[:5] ==  'Error':
                    print(answer)
                    time.sleep(120)
                    return create_email_json()
                
                answer = answer.strip()

                new_json.append({
                    "holiday_name": holiday["holiday_name"],
                    "holiday_email": answer
                })

                json_object = json.dumps(new_json, indent=4)
                with open("json/new_holiday_emails.json", "w") as outfile:
                    outfile.write(json_object)

                if len(new_json) == 366:
                    return print('Success 200: All email blurbs created!')

                print(f'{len(new_json)} / 366')
                time.sleep(60)

create_email_json()
