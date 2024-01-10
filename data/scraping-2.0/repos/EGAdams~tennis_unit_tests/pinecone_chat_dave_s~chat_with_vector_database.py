import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
import pinecone

TURBO_16K_MODEL     = "gpt-3.5-turbo-16k"
TURBO_MODEL         = "gpt-3.5-turbo"
CURRENT_DIRECTORY   = "/home/adamsl/linuxBash/SMOL_AI/tennis_unit_tests/pinecone_chat_dave_s/"

def open_file(filepath):
    with open( CURRENT_DIRECTORY + filepath, 'r', encoding='utf-8' ) as infile:
        return infile.read()


def save_file(filepath, content ):
    with open( CURRENT_DIRECTORY + filepath, 'w', encoding='utf-8' ) as outfile:
        outfile.write(content )


def load_json(filepath):
    with open( CURRENT_DIRECTORY + filepath, 'r', encoding='utf-8' ) as infile:
        return json.load( infile )


def save_json( filepath, payload):
    with open( CURRENT_DIRECTORY + filepath, 'w', encoding='utf-8' ) as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime( unix_time ):
    return datetime.datetime.fromtimestamp( unix_time ).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content, engine='text-embedding-ada-002' ):
    content = content.encode(encoding='ASCII',errors='ignore' ).decode()  # fix any UNICODE errors
    response = openai.Embedding.create( input=content,engine=engine )
    vector = response[ 'data' ][ 0 ][ 'embedding' ]  # this is a normal list
    return vector



def ai_completion(prompt, engine=TURBO_MODEL, temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0):
    max_retry = 5
    retry = 0
    prompt = prompt.encode( encoding='ASCII', errors='ignore').decode()  # fix any UNICODE errors
    
    # Constructing the messages list for the chat API based on the prompt.
    # This is a basic example; you might need to adjust the list depending on your specific requirements.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    print( 'messages:', messages )
    answ = input( "press enter to continue, x to exit: $" )
    if answ == "x":
        exit()
    elif answ == "":
        pass
    
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen
            )
            text = response[ 'choices' ][ 0 ][ 'message' ][ 'content' ].strip()
            # of os is windows, clean up the text
            if os.name == 'nt':
                # clean up the text for Windows
                text = re.sub(r'[\\r\\n]+', '\\n', text)
                text = re.sub(r'[\\t ]+', ' ', text)
            else:
                # clean up the text for Linux
                text = re.sub(r'[\r\n]+', '\n', text)
                text = re.sub(r'[\t ]+', ' ', text)
            
            # Optional: saving the response to a log file (retained from the original function)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists( 'gpt3_logs' ):
                os.makedirs( 'gpt3_logs' )
            save_file( 'gpt3_logs/%s' % filename, prompt + '\\n\\n==========\\n\\n' + text )
            
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print( 'Error communicating with OpenAI:', oops )
            sleep( 1 )


def load_conversation( results_arg ):  # comes from:  vdb.query( vector = embedded_user_input, top_k = convo_length )
    result = list()
    for matching_unique_id in results_arg[ 'matches' ]:
        filename = 'nexus/%s.json' % matching_unique_id[ 'id' ]
        # if filename exists, load it and append it to the result list, otherwise skip it
        if not os.path.exists( filename ):
            print ( 'file not found:', filename )
            continue
        else:
            print ( 'file found:', filename )
        info = load_json( 'nexus/%s.json' % matching_unique_id[ 'id' ])
        result.append( info )
    ordered = sorted( result, key=lambda d: d[ 'time' ], reverse=False )  # sort them all chronologically
    messages = [ i[ 'message' ] for i in ordered ]
    return '\n'.join( messages ).strip()


if __name__ == '__main__':
    #get the current working directory
    
    convo_length = 30
    openai.api_key = open_file( 'key_openai.txt' )
    pinecone.init( api_key=open_file( 'key_pinecone.txt' ), environment='northamerica-northeast1-gcp' )
    vdb = pinecone.Index( "debug-memory" )
    while True:
        print( '\n\n' )
        print( '      ///////////////////////////////////////////' )
        print( '      Welcome to the main LLM Pinecone Chatbot'    )
        print( '      ///////////////////////////////////////////' )
        print( '\n\n' )
        print( '1. input prompt.md' )
        print( '2. input user input' )
        print( '3. exit' )
        print( '\n\n' )
        choice = input( 'Please select an option: ' )
        if choice == "1":
            print( "using prompt.md, ok? <enter> to continue.  ctrl-c to quit" )
            user_input = open_file( 'prompt.md' )
        elif choice == "2":
            user_input = input( '\n\nUSER: ' )
        elif choice == "3":
            print( "Goodbye!" )
            exit()
        else:
            print( 'invalid choice' )
            continue  
        ###
        data_for_pinecone_upsert = list()   # initialize the list that will ultimately be 
        timestamp = time()                  # upserted to the vector database
        timestring = timestamp_to_datetime( timestamp )
        embedded_user_input = gpt3_embedding( user_input )
        unique_id = str( uuid4())
        metadata = { 'speaker': 'USER', 'time': timestamp, 'message': user_input, 'timestring': timestring, 'uuid': unique_id }
        save_json( 'nexus/%s.json' % unique_id, metadata ) # <<--- save to .json on local ---<<<
        data_for_pinecone_upsert.append(( unique_id, embedded_user_input ))  # <<--- this data is going to pinecone vdb ---<<<
        ###
        ###  Now we have the user input not only saved to our local file, but it is also placed in the built-in mutable
        ###  sequence that we will ultimately be inserted into the vector database under the same unique_id.
        ###
        results = vdb.query( vector=embedded_user_input, top_k=convo_length )# search for relevant message unique ids in vsd
        conversation = load_conversation( results )  # with these unique ids, which where very cheap to aquire, we load the
                                                     # relevant conversation data from our local file system
        prompt = open_file( 'prompt_response.txt' ).replace( '<<CONVERSATION>>', conversation ).replace( '<<MESSAGE>>', user_input )
        ###
        ai_completion_text = ai_completion( prompt )  # <<-- send the prompt created from the template to the model ---<<<
        timestamp = time()
        timestring = timestamp_to_datetime( timestamp )
        embedded_ai_completion = gpt3_embedding( ai_completion_text )
        unique_id = str( uuid4())
        metadata = { 'speaker': 'RAVEN', 'time': timestamp, 'message': ai_completion_text,
                     'timestring': timestring, 'uuid': unique_id }
        ###
        save_json( 'nexus/%s.json' % unique_id, metadata ) # <<--- save ai answer to a .json file ---<<<
        ###
        data_for_pinecone_upsert.append(( unique_id, embedded_ai_completion )) # <--- add ai answer to data to be upserted ---<<<
        vdb.upsert( data_for_pinecone_upsert ) # <----------------------------------- upsert the data to pinecone ------------<<<
        print( '\n\nRAVEN: %s' % ai_completion_text )  
        
        # just noticed that the unique_id is being used in the upsert and
        # the save_json, so it's being saved twice. How are we referencing both?
        # we make a unique id for the data here on our file system.  the vector database
        # returns the unique_ids that have relevance.  we use those ids to get the relevant data from
        # our file system.  the unique_id from the vector database links to the unique id on the local file
        # system that has the complete relevant information.