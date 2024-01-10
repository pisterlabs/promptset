
#
# https://chat.openai.com/c/59bec3a5-5066-4d85-bc6c-ea483a9b381c
#
import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time, sleep
from uuid import uuid4
import datetime
import pinecone
from tree_sitter import Language, Parser

TURBO_16K_MODEL = "gpt-3.5-turbo-16k"
TURBO_MODEL = "gpt-3.5-turbo"
CURRENT_DIRECTORY = "/home/adamsl/linuxBash/agents/pinecone_chat_dave_s/"
CACHED_EMBEDDINGS_PATH = CURRENT_DIRECTORY + "cached_embeddings/"

# GO_LANGUAGE = (Language('build/my-languages.so', 'go'), "*.go")
# JS_LANGUAGE = (Language('build/my-languages.so', 'javascript'), "*.js")
# PY_LANGUAGE = (Language('build/my-languages.so', 'python'), "*.py")
# SWIFT_LANGUAGE = (Language('build/my-languages.so', 'swift'), "*.swift")
# JAVA_LANGUAGE  = (Language('build/my-languages.so', 'java'), "*.java")
CPP_LANGUAGE   = (Language('/home/adamsl/linuxBash/agents/pinecone_chat_dave_s/build/my-languages.so', 'cpp'), "*.cpp")

current_language = CPP_LANGUAGE


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

# File Operations
def open_file(filepath):
    with open(CURRENT_DIRECTORY + filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(CURRENT_DIRECTORY + filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def load_json(filepath):
    with open(CURRENT_DIRECTORY + filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)

def save_json(filepath, payload):
    with open(CURRENT_DIRECTORY + filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)

# Utilities
def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector

def ai_completion(prompt, engine=TURBO_MODEL, temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()  # fix any UNICODE errors
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(messages=messages, model=engine, temperature=temp, max_tokens=tokens, top_p=top_p, frequency_penalty=freq_pen, presence_penalty=pres_pen)
    return response.choices[0].message['content']

def get_functions(filepath):
    codestr = open(filepath).read().replace("\r", "\n")
    parser = Parser()
    # parser.set_language( "cpp" ) # TODO: take out hard coded language
    parser.set_language( current_language[ 0 ])
    tree = parser.parse(bytes(codestr, "utf8"))
    cursor = tree.walk()
    cursor.goto_first_child()
    functions = []
    while True:
        print( "cursor.node.type: ", cursor.node.type )
        print( "cursor.node: ", cursor.node.text )
        print( "cursor.goto_next_sibling(): ", cursor.goto_next_sibling() )
        print( "cursor.goto_first_child(): ", cursor.goto_first_child() )
        print( "cursor.goto_parent(): ", cursor.goto_parent() )
        if cursor.node.type == 'function_definition':
            functions.append(codestr[cursor.node.start_byte:cursor.node.end_byte])
        if not cursor.goto_next_sibling():
            break
    return functions

def generate_embeddings_for_code_chunks(code_chunks):
    embeddings = []
    for chunk in code_chunks:
        embeddings.append(gpt3_embedding(chunk))
    return embeddings

def save_code_chunks_and_embeddings(chunks, embeddings, directory="nexus"):
    for chunk, embedding in zip(chunks, embeddings):
        unique_id = str(uuid4())
        metadata = {'code': chunk, 'embedding': embedding, 'uuid': unique_id}
        save_json(f'{directory}/{unique_id}.json', metadata)

# Main Functionality
def main():
    # Initialize Pinecone connection (assuming you have a namespace/environment setup)
    openai.api_key = open_file( 'key_openai.txt' )
    thePineconeKey = open_file( 'key_pinecone.txt' )
    pinecone.init( thePineconeKey, environment='northamerica-northeast1-gcp' )
    vdb = pinecone.Index( "debug-memory" )
    convo_length = 30
    while True:
        print("Choose an option:")
        print("1. Run a query from a prompt stored in a file.")
        print("2. Run a query from the command line.")
        print("3. Read an entire codebase.")
        print("4. fire up the Plan and Execute Agent.")
        print("x. Exit.")
        choice = input("Enter your choice: ")
        if choice == "1":
            # TODO: Handle option 1
            pass
        elif choice == "2":
            user_input = input( '\n\nUSER: ' )
            pass
        elif choice == "3":
            if os.path.exists( CACHED_EMBEDDINGS_PATH ):
                with open("previous_search_path.txt", "r") as file:
                    code_root = file.read()
            else:
                print("Please type in the path to the code directory you want to search/embed/query:")
                code_root = input()
                if code_root.strip() == "":
                    code_root = os.getcwd()
                with open("previous_search_path.txt", "w") as file:
                    file.write( code_root )
                    
            # TODO: take out hard coded path
            code_root = "/home/adamsl/linuxBash/SMOL_AI/tennis_unit_tests/Mode1Score/Mode1Score.cpp"
            code_chunks = get_functions( code_root )
            embeddings = generate_embeddings_for_code_chunks(code_chunks)
            save_code_chunks_and_embeddings(code_chunks, embeddings)
            
            # Saving and upserting to Pinecone
            for chunk, embedding in zip(code_chunks, embeddings):
                # Generate a unique ID for each code chunk
                unique_id = str(uuid4())
                
                # Save the code chunk and its embedding to a .json file in the nexus directory
                metadata = {'code': chunk, 'embedding': embedding, 'uuid': unique_id}
                save_json(f'nexus/{unique_id}.json', metadata)
                
                # Upsert the chunk and its embedding to Pinecone
                # pinecone.upsert(items={unique_id: embedding})
            
            # Close the Pinecone connection... maybe not necessary?
            
            print( "Codebase read, chunks saved and embeddings generated." )
            continue
        
        elif choice == "4":
            print( "initializing Plan and Execute Agent..." )
            
            break
        
        elif choice == "x":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
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

if __name__ == "__main__":
    main()



# Tests for file operations
def test_file_operations():
    # Mock data for testing
    mock_json_data = {
        "name": "ChatGPT",
        "description": "A friendly chatbot"
    }
    # File paths for testing
    test_json_filepath = "/mnt/data/test_file_operations.json"
    
    # Test save_json and load_json
    save_json(test_json_filepath, mock_json_data)
    loaded_data = load_json(test_json_filepath)
    assert loaded_data == mock_json_data, f"Expected {mock_json_data} but got {loaded_data}"
    
    # Test open_file and save_file
    save_file(test_json_filepath, "Test content")
    loaded_content = open_file(test_json_filepath)
    assert loaded_content == "Test content", f"Expected 'Test content' but got {loaded_content}"
    
    print("All file operations tests passed!")

# Uncomment the following line to run the file operations tests
# test_file_operations()
