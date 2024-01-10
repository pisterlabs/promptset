import requests
import geocoder
import math
import pandas as pd
import cohere

API_KEY="d2bb4999c6964195b383526d9412b5c8" 
# replace with your API token
base_url = "https://api.assemblyai.com/v2"

def get_coords(address):
    g = geocoder.bing(address, key='Aowcdh3tB--xi-HGt95MZr7jCFWqDenSzKp0yDtC2AgfH_HstHkEBY2XkFgw9XW9')
    return [g.json['lat'], g.json['lng']]

def get_address(transcription_id):
    answer = question(transcription_id, q_format("Extract the full address or location mentioned in the transcript ", "One line"))
    address = answer["response"][0]["answer"]
    return address

def q_format(prompt, format):
    questions = [
    
    {
        "question": f"{prompt}",
        "answer_format": f'''{format}
        '''
    }
]
    return questions

def post_lemur(api_token, transcript_ids, questions=None, type='qa', data={}):
    if type=='qa':
        url = "https://api.assemblyai.com/lemur/v3/generate/question-answer"
    else:
        url = "https://api.assemblyai.com/lemur/v3/generate/summary"

    headers = {
        "authorization": api_token
    }
    if not questions and not data:
        data = {
        "transcript_ids": transcript_ids,
        "model": "basic"
    }   
    else:

        data = {
            "transcript_ids": transcript_ids,
            "questions": questions,
            "model": "basic"
        }

    response = requests.post(url, json=data, headers=headers)
    return response
def question(transcript_id,question):
    lemur_output = post_lemur(API_KEY, [transcript_id], question)
    lemur_response = lemur_output.json()

    if "error" in lemur_response:
        print(f"Error: { lemur_response['error'] }")
    else:
        return(lemur_response)

def get_nearest(type, lat, long):
    print(type)
    if type=="Law and Order":
        police_db = pd.read_csv("hyd_police_stn_jurisdictions.csv")
        nearest=[]
        count=0
        for index, entry in police_db.iterrows():
            distance = 3959 * math.acos( math.cos( math.radians(lat) ) * math.cos( math.radians( float(entry["Y"]) ) ) * 
math.cos( math.radians( long ) - math.radians(float(entry["X"])) ) + math.sin( float(math.radians(entry["Y"] )) ) * 
math.sin( math.radians( lat ) ) )
            if distance < 15:
                nearest.append([list(entry), distance])
                count +=1
            if count==3:
                break
            
        return sorted(nearest, key=lambda x: x[1])
    if type=="Fire" or "Natural Disaster":
        fire_db = pd.read_csv("hyderabad fire stations.csv")
        nearest=[]
        count=0
        for index, entry in fire_db.iterrows():
            distance = 3959 * math.acos( math.cos( math.radians(lat) ) * math.cos( math.radians( float(entry["Y"]) ) ) * 
math.cos( math.radians( long ) - math.radians(float(entry["X"])) ) + math.sin( float(math.radians(entry["Y"] )) ) * 
math.sin( math.radians( lat ) ) )
            if distance < 15:
                nearest.append([list(entry), distance])
                count +=1
            if count==3:
                break
        
        
        return sorted(nearest, key=lambda x: x[1])

def get_category(transcription):
    concatenated_text = '\n'.join([f"{item[0]}: {item[1]}" if item[0] != 'You.' else item[1] for item in transcription])
    co = cohere.Client('9gTWsgGsGUoSSKzUvLuZdcuEtuBO2CIhiG9s17nU') # This is your trial API key
    response = co.classify(
    model='2196d10d-e411-417d-b342-2882c65248f5-ft',
    inputs=[concatenated_text ],
    )
    return(response.classifications[0].prediction)

def get_severity(transcription_id):
    severity = question(transcription_id, q_format("Determine how severe the emergency is, with high level destruction being 10 while a very small incident is 1", "floating point number between 1-10"))
    return severity["response"][0]["answer"]