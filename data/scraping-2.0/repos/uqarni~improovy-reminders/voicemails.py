# import redis
# import os
# import platform
import openai
from openai_client import generate_response
import time
from pipedrive import PipedriveClient
from justcall import send_text
from datetime import datetime, timedelta
from supabase_client import SupabaseClient
import os
import re

#improovy justcall info
improovy_api_key = "a475b0ecf1d1ba78ec7a9bc49d60f225531f3617"
improovy_api_secret = "bed00ba3e48de573ab7841e11971c32948edff6f"

# test = send_text('+16084205020', '+17736206534', 'testing1234', improovy_api_key, improovy_api_secret)
# if test == {'status': 'success'}:
#     print(True)

import requests

def query_calls(api_key, api_secret, per_page=1):
    headers = {
        "Accept": "application/json",
        "Authorization": f"{api_key}:{api_secret}"
    }

    data = {
        "type": '5',
        "per_page": per_page
    }

    url = "https://api.justcall.io/v1/calls/query"

    response = requests.post(url, headers=headers, json=data)
    return response.json()



def get_transcription(api_key, api_secret, id):
    headers = {
        "Accept": "application/json",
        "Authorization": f"{api_key}:{api_secret}"
    }

    params = {
        "id": id,
        "platform": 1
    }

    url = "https://api.justcall.io/v1/justcalliq/transcription"

    response = requests.get(url, headers=headers, json=params)
    return response.json()


def download_call_recording(api_key, api_secret, call_id, filename="recording.mp3"):
    headers = {
        "Accept": "application/json",
        "Authorization": f"{api_key}:{api_secret}"
    }

    data = {
        "id": call_id
    }

    url = "https://api.justcall.io/v1/calls/download"
    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Error {response.status_code}: {response.text}")

def add_voicemails():
    voicemail_body = query_calls(improovy_api_key, improovy_api_secret, 10)

    voicemails = voicemail_body['data']
    print('number of voicemails retrieved: ' + str(len(voicemails)))
    current_time = datetime.utcnow()
    time_limit = current_time - timedelta(minutes = 10)

    filtered_voicemails = []

    for voicemail in voicemails:
        voicemail_time_str = voicemail['time_utc']
        voicemail_time = datetime.strptime(voicemail_time_str, '%Y-%m-%d %H:%M:%S')

        if voicemail_time > time_limit:
            filtered_voicemails.append(voicemail)

    # Now `filtered_voicemails` contains voicemails from the last 25 hours
    # You can iterate through it the same way as the original list


    count = 1
    print('number of voicemails in the last 10 minutes: ' + str(len(filtered_voicemails)))
    for voicemail in filtered_voicemails:
        print('pulling voicemail ' + str(count))
        print('voicemail ' + str(count) + ' pulled')
        url = voicemail['recording']
        response = requests.get(url)
        with open('downloaded_file' + str(count) + '.mp3', 'wb') as file:
            file.write(response.content)

        audio_file= open('downloaded_file' + str(count) + '.mp3', "rb")
        print('generating transcript for voicemail ' + str(count))
        print('voicemail raw contents: ', str(voicemail))

        try:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        except:
            transcript = {'text': 'text too short'}
        print(transcript['text'])
        print('#####')

        count += 1
        crm = PipedriveClient('0a267f5c893e31a9dca5ff9bb3d0397266f69443')
        contactids = crm.contactIDs_from_num(voicemail['contact_number'])
        dealids = []
        for contactid in contactids:
            print('personid')
            print(contactid)
            deals = crm.deals_from_personID(contactid)
            dealids = []
            for deal in deals.get('data', []):
                dealids.append(deal['id'])
            add_note = crm.add_note(content = 'VOICEMAIL: ' + transcript['text'], personid = contactid)
            print('added note' + str(add_note))

        for dealid in dealids:
            print('dealid')
            print(dealid)
            add_note = crm.add_note(content = 'VOICEMAIL: ' + transcript['text'], dealid = dealid)
            print('added note ' + str(add_note))

        try:
            #VOICEMAIL RESPONDER STARTS HERE
            #check to see if weve already sent a voicemail to this person
            contact_number = '+' + voicemail['contact_number']
            print('contact_number: ', contact_number)
            
            db = SupabaseClient()

            db_contact = db.fetch_by_contact_phone_and_orgid('contacts', contact_number, 'improovy')
            print('db_contact: ', db_contact)

            if db_contact == None:
                #create contact in supabase
                print('creating contact in supabase')
                new_contact = {
                    'org_id': 'improovy',
                    'last_contact': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'contact_phone': contact_number,
                    'custom_data': '\{\}'
                }
                db.insert('contacts', new_contact)
                db_contact = db.fetch_by_contact_phone_and_orgid('contacts', contact_number, 'improovy')

            if db_contact.get('group','N/A') == 'voicemail':
                print('already sent a voicemail to this person')
                return 200
            else:
                id = db_contact.get('id')
                db.update('contacts', {'group': 'voicemail'}, id)
                print('updated contact group to voicemail')
                
            summarizer_prompt = db.get_system_prompt_prod("bots", "mike_voicemail_summarizer")

            summarizer_prompt = summarizer_prompt.format(voicemail = transcript['text'], name="Mike")
            summarizer_prompt = [{"role": "system", "content": summarizer_prompt}]

            ## generate summary
            initial_message, prompt_tokens, completion_tokens = generate_response(summarizer_prompt)
            print('initial message: ', initial_message, ' to contact: ', contact_number)

            #parse initial message for snippet
            project_description_pattern = r"voicemail about (.+?)\. Can"
            initial_text_list = [initial_message]
            project_description = [re.search(project_description_pattern, initial_message).group(1) for i in initial_text_list]

            #store project description in custom data
            custom_data = db_contact.get('custom_data')
            custom_data['project_description'] = project_description
            custom_data['voicemail'] = transcript['text']
            db.update_contact(id, {'custom_data': custom_data})

            #send initial message using bot
        except Exception as e:
            print('error: ', e)
            return 200

        

        



