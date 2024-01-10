import hashlib
import base64
import urllib.parse
import json
import os
import time

secret_value = os.environ.get('SERVICEACCOUNTKEY')
secret_value = json.loads(secret_value)



def generate_code_challenge(code_verifier):
    sha256 = hashlib.sha256()
    sha256.update(code_verifier.encode('utf-8'))
    code_challenge = base64.urlsafe_b64encode(sha256.digest()).rstrip(b'=').decode('utf-8')
    return code_challenge
def create_auth_url(*,client_id, scope, code_challenge, state):
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": "https://localhost",
        "scope": scope,
        "state" : state,
        "code_challenge": code_challenge,
        "code_challenge_method" : "S256"              
    }    
    encoded_params = urllib.parse.urlencode(params)
    authorization_url = f"https://twitter.com/i/oauth2/authorize?{encoded_params}"
    return authorization_url
def code_extract(url):
    parsed_url = urllib.parse.urlparse(url)
    query_string_dict = urllib.parse.parse_qs(parsed_url.query)
    code_extracted = query_string_dict.get("code", [None])[0]
    return code_extracted       
def database_retriever(*,data):
    import firebase_admin
    from firebase_admin import credentials, db 
    accepted_values = ["Auth_Para", "Client", "Consumer", "Developer Access", "bot_token"]
    cred = credentials.Certificate(secret_value)
    database_url = secret_value["databaseURL"]
    try:
        firebase_admin.initialize_app(cred, {
            'databaseURL': database_url})
    except ValueError:
        pass       
    ref = db.reference('Twitter/')
    users_ref = ref.child('Tokens and Secrets')
    info = users_ref.get()
    list_database = []
    for k,v in info.items():
        list_database.append( (k,v) )  
    index = accepted_values.index(data)
    if index == -1:
        raise ValueError(f"Invalid database name. Accepted values: {accepted_values}")    
    return list_database[index][1]                   
def update_token(*,New_Access_token, New_Refresh_Token):
    import firebase_admin
    from firebase_admin import credentials, db 
    cred = credentials.Certificate(secret_value)
    database_url = secret_value["databaseURL"]
    try:
        firebase_admin.initialize_app(cred, {
            'databaseURL': database_url})
    except ValueError:
        pass
    ref = db.reference('Twitter/')
    users_ref = ref.child('Tokens and Secrets')
    hopper_ref = users_ref.child('bot_token') 
    hopper_ref.update({
        'Access_Token' : New_Access_token,
        'Refresh_Token' : New_Refresh_Token})
def generateTLDR(*, prompt):
    import openai
    import firebase_admin
    from firebase_admin import credentials, db 
    cred = credentials.Certificate(secret_value)
    database_url = secret_value["databaseURL"]
    try:
        firebase_admin.initialize_app(cred, {
            'databaseURL': database_url})
    except ValueError:
        pass   
    ref = db.reference('OpenAI/')
    ChatGPT_ref = ref.child('ChatGPT')  
    # Initialize the API key
    openai.api_key = ChatGPT_ref.get()['API']
    # Generate the summary
    completions = openai.Completion.create(
        engine='text-babbage-001',
        prompt=f'''Summarize the News Article:\n\{prompt}\n''',
        max_tokens=400,
        n=1,
        stop=None,
        temperature=0.85 )
    summary = completions.choices[0].text
    return summary
def NewsStorer(*, date,title,link,content,count,tldr):
    import firebase_admin
    from firebase_admin import credentials, db 
    cred = credentials.Certificate(secret_value)
    database_url = secret_value["databaseURL"]
    try:
        firebase_admin.initialize_app(cred, {
            'databaseURL': database_url})
    except ValueError:
        pass    
    ref = db.reference('News/')
    Source_ref = ref.child(f'Source {count}')
    Source_ref.set({
    'title' : title,
    'content': content,
    'link': link,
    'date': date ,
    'tldr': tldr 
    })   

def NewsLooper():
    import firebase_admin
    from firebase_admin import credentials, db 
    cred = credentials.Certificate(secret_value)
    database_url = secret_value["databaseURL"]
    try:
        firebase_admin.initialize_app(cred, {
            'databaseURL': database_url})
    except ValueError:
        pass   
    tweet = None
    while True:
        ref = db.reference('News/')
        News = ref.get()        
        for Source in News:
            title = News[Source]['title']    
            if not title == 'Tweeted':
                tweet = News[Source]['tldr']
                link = News[Source]['link']          
                Source_ref = ref.child(Source)
                Source_ref.update({'title': 'Tweeted'})
                return tweet, link  
        if tweet == None:
            print('Time to generate new articles')
            exec(open("Source/NewsData.py").read())
            time.sleep(5)
            print('Generated')
            continue
    
    
    
    
    
    
    
    

    
    
    
    



    
    
