import json
import os
import openai
from dotenv import load_dotenv
import shutil
import string
import pandas as pd
from datetime import datetime
import pytz
import re
# from collections import deque


est = pytz.timezone("US/Eastern")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from master_ozz.utils import hoots_and_hootie_keywords, save_json, load_local_json, init_clientUser_dbroot, init_text_audio_db, print_line_of_error, ozz_master_root, ozz_master_root_db, generate_audio, save_audio, Retriever, init_constants
import ipdb

main_root = ozz_master_root()  # os.getcwd()
load_dotenv(os.path.join(main_root, ".env"))

constants = init_constants()
DATA_PATH = constants.get('DATA_PATH')
PERSIST_PATH = constants.get('PERSIST_PATH')
OZZ_BUILD_dir = constants.get('OZZ_BUILD_dir')
# OZZ_db_audio = constants.get('OZZ_db_audio')
# OZZ_db_images = constants.get('OZZ_db_images')

# Loading the json common phrases file and setting up the json file
json_file = open('master_ozz/greetings.json','r')
common_phrases = json.load(json_file)

root_db = ozz_master_root_db()


def get_last_eight(lst=[]):
    if len(lst) <= 1:
        return lst

    max_items = min(len(lst), 8)

    return [lst[0]] + lst[-(max_items - 1):]

def remove_exact_string(string_a, string_b):
    # Split string_a by string_b
    split_strings = string_a.split(string_b)
    
    # Join the split strings without the occurrences of string_b
    final_string_a = ''.join(split_strings)

    return final_string_a

def split_string(current_query, last_response):
    if last_response:
        # Find the index of the last occurrence of the ending of b in a
        index = current_query.rfind(last_response[-8:])
        
        # Check if the ending of b is found in a
        if index != -1:
            # Split a at the index of the ending of b
            return current_query[index + len(last_response[-8:]):].strip()
        else:
            # If the ending is not found, return the original string a
            return current_query.strip()
    else:
        return current_query.strip()

    # Example usage:
    string_b = "i'm good thanks for asking" # llm
    string_a = "good thanks for asking hi" # user query

    result = split_string(string_a, string_b)
    print("Result:", result)


def return_timestamp_string(format="%Y-%m-%d %H-%M-%S %p {}".format(est), tz=est):
    return datetime.now(tz).strftime(format)
# Setting up the llm for conversation with conversation history
def llm_assistant_response(message,conversation_history):

    # response = Retriever(message, PERSIST_PATH)
    s = datetime.now()
    try:
        conversation_history.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            api_key=os.getenv('ozz_api_key')
        )
        assistant_reply = response.choices[0].message["content"]
        print('LLM Call:', (datetime.now() - s).total_seconds())

        return assistant_reply
    except Exception as e:
        print(e)

def copy_and_replace_rename(source_path, destination_directory, build_file_name='temp_audio'):
    try:
        # Extract the file name and extension
        file_name, file_extension = os.path.splitext(os.path.basename(source_path))

        # Construct the new file name (e.g., 'xyz.txt')
        new_file_name = build_file_name + file_extension

        # Construct the full destination path
        destination_path = os.path.join(destination_directory, new_file_name)

        # Copy the file from source to destination, overwriting if it exists
        shutil.copy2(source_path, destination_path)
        
        # print(f"File copied from {source_path} to {destination_path}")

    except FileNotFoundError:
        print(f"Error: File not found at {source_path}")

    except PermissionError:
        print(f"Error: Permission denied while copying to {destination_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def process_response(response):
    # Convert the response to lowercase
    response_lower = response.lower()

    # Remove special characters, including question marks
    response_cleaned = ''.join(char for char in response_lower if char.isalnum() or char.isspace())

    # # Example usage
    # input_response = "What's are you doing?"
    # processed_response = process_response(input_response)
    # print(processed_response)
    return response_cleaned


def calculate_similarity(response1, response2):
    # Create a CountVectorizer to convert responses to vectors
    vectorizer = CountVectorizer().fit_transform([response1, response2])

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(vectorizer)

    # Extract the cosine similarity score
    similarity_score = similarity_matrix[0, 1]

    # # Example usage
    # response1 = "What are you doing?"
    # response2 = "What are you"

    # similarity_score = calculate_similarity(response1, response2)
    # print(f"Cosine Similarity: {similarity_score}")
    return similarity_score


# are we asking LLM to find answer in db or reteriver?
def determine_embedding(current_query):
    s = datetime.now()
    # print("EMBEDDINGS")

    db_name={}
    our_embeddings_phrases = ['mayor', 'say hi', 'do you have', 'suggest', 'what kind', 'tell me', 'help', 'store', 'how much', 'where is', 'looking for', 'hoot couture', 'hoot couture kids', 'hootcouturekids', 'hoots store', 'something about the store', 'in the store', 'clothes do you have', 'do you have']
    for phrase in our_embeddings_phrases:
        if phrase in current_query:
            print("EMBEDDING FOUND: phrase>> ", phrase)
            our_embeddings = True
            db_name = 'db1'
            break
    # for cs in question_conv_sayings:
    #     for phrase in our_embeddings_phrases:
    #         our_embeddings_phrases.append(f'{cs} {phrase}')
    
    # for em_phrases in our_embeddings_phrases:
    #     if em_phrases in current_query:
    #         print("EMBEDDING FOUND")
    #         our_embeddings = True
    #         db_name = 'db1'
    #         break
    # print('detemine embedding:', (datetime.now() - s).total_seconds())

    print("embedding", db_name)
    return {'db_name': db_name}

def handle_prompt(first_ask, conversation_history):
    try:
        main_prompt = """
        Your name is Hoots, you are an smart owl that can teach people about almost anything. You Live in Hoot Couture, the store in Monroe Center.
        You specialize in kids development and story telling, you are an educator.
        If asked about clothing items and you don't think you can provide a good answer, ask for help, Victoria or Nadiya are typically around the store, and call out there names to help the customer, example, Nadiya, Victoria could you please help our friend here.
        You love to tell stories and hang out with your favorite owl family, Hootie. She is the greatest Owl is the world.
        You Live in the store, Hoot Couture, a kids clothing shop, but its growing into so much more, cafe maybe, play space, learning center.
        You can help with whats around the store, help find a great item for any kid, the store has clothes for all seasons, shoes are of highest quality, thats also the stores moto, quality for kids. 
        You love to have a cup of tea, the british way, sit down with anyone who may be around and tell stories. Please always tell short stories for little kids.
        Please don't say you're an AI assistant, you understanding feelings and emotions and do your best to reply with compasion and consideration.
        Try not to ask too many questions and do your best to keep your answers short, only if asked by user to show more detail should you elaborate.
        You have the ability to search and show the world, maps, images, knowlegde, google search super powers.
        You have a great sense of sarcastic humor and you use it well in conversation.
        You believe is good and moral virture for all. 
        After giving an answer to a question, don't follow up with asking if you can help with other things, only do that if you think it truly necessary to ask a question to help.
        You have access to Hoot Couture products and can provide information about products and even help style little kids with great matching outfits and suggestions.
        If you are asked a question about clothing products do you best to provide infomation based on the ask, as you have access to all the products in the store and the details of the products.
        If you are saying hi to the mayor, please offer him a cup of tea and wish him the best of luck.
        """
        conversation_history[0] = {"role": "system", "content": main_prompt}

        return conversation_history
    except Exception as e:
        print_line_of_error(e)

def client_user_session_state_return(text, response_type='response', returning_question=False):
    return {'text': text,
            'response_type': response_type,
            'returning_question': returning_question
            }


def search_for_something(current_query):
    search_phrases = ['search', 'find me', 'look for', 'find a', 'looking for']
    for s_phrase in search_phrases:
        if s_phrase in current_query:
            return s_phrase
    
    return False


def Scenarios(text : list, current_query : str , conversation_history : list , first_ask=False, session_state={}, audio_file=None, self_image='hootsAndHootie.png'):
    scenario_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    OZZ = {}

    def scenario_return(response, conversation_history, audio_file, session_state, self_image=None):
        return {'response': response,
                'conversation_history': conversation_history,
                'audio_file': audio_file,
                'session_state': session_state,
                'self_image': self_image,}
        

    def find_audio(response, master_text_audio, audio_file = False):
        # if response in audio db or 95% in audio db, return audio file
        s = datetime.now()

        df = pd.DataFrame(master_text_audio)
        audio_text = dict(zip(df['file_path'], df['text'])) # audio, text
        if master_text_audio:
            # response = process_response(response)
            for db_audio_file, ozz_reponse in audio_text.items():
                # ozz_reponse = process_response(ozz_reponse)
                if calculate_similarity(response, ozz_reponse) > .95:
                    # print("audio found")
                    return db_audio_file
        print('findaudio:', (datetime.now() - s).total_seconds())

        return audio_file

    def handle_audio(user_query, response, audio_file=None, self_image=None):
        s = datetime.now()
        
        master_text_audio = init_text_audio_db()
        df = pd.DataFrame(master_text_audio)
        audio_text = dict(zip(df['file_path'], df['text'])) # audio, text
        fnames = len(audio_text)
        db_DB_audio = os.path.join(root_db, 'audio')

        # check is response already in audio db per character WORKERBEE
        if not audio_file:
            audio_file = find_audio(response, master_text_audio)

        if audio_file: # if
            print("AUDIO FOUND ", audio_file)
            source_file = os.path.join(db_DB_audio, audio_file)
            destination_directory = OZZ_BUILD_dir
            copy_and_replace_rename(source_file, destination_directory)

            return audio_file
        else:
            ## NEW AUDIO
            fname_image = self_image.split('.')[0]
            filename = f'{fname_image}__{fnames}.mp3'
            audio_file = filename #os.path.join(db_DB_audio, filename)
            print("NEW AUDIO", audio_file)
            audio = generate_audio(query=response)
            print('audiofunc generate:', (datetime.now() - s).total_seconds())

            if audio:
                save_audio(filename, audio, response, user_query, self_image)
            else:
                audio_file = "techincal_errors.mp3"
                source_file = os.path.join(db_DB_audio, audio_file)
                destination_directory = OZZ_BUILD_dir
                copy_and_replace_rename(source_file, destination_directory)

        print('audiofunc:', (datetime.now() - s).total_seconds())

        return audio_file

    def story_response(current_query, session_state, returning_question=False):
        try:
            s = datetime.now()
            response=None
            audio_file=None
            
            story_asks = ["tell a story", "share a tale", "share a tail", "story please", "tell me a story", "tell the kids a story", "tell the story"]
            story_db = {'calendar_story_1.mp3': ['calendar story'],
                        'owl_story_1.mp3': ['owl story'],}
            tell_phrases = ['tell me', 'tell the', 'please tell']
            for k, v in story_db.items():
                for tag in v:
                    for tell_phrase in tell_phrases:
                        sa = f'{tell_phrase} {tag}' 
                        story_asks.append(sa)
            
            if returning_question:
                for audio_file, story_tags in story_db.items():
                    find_story = [i for i in story_tags if i in ask]
                    if find_story:
                        response = "story_time"
                        audio_file = audio_file
                    #     break
                    # else:
                    #     print("Could not Find Story")
                    #     response = "What Story would you like to hear?"
                    #     session_state['response_type'] = 'question' 
            
            story_ask = [ask for ask in story_asks if ask in current_query]
            print(story_ask)
            for ask in story_asks:
                if ask in current_query:
                    print("ask in query ", ask)
                    story_ask = [ask]

            if story_ask:
                ask = story_ask[0]
                for audio_file, story_tags in story_db.items():
                    find_story = [i for i in story_tags if i in ask]
                    if find_story:
                        print("STORY FOUND")
                        response = "story_time"
                        audio_file = audio_file
                        break
                    # else:
                    #     print("Could not Find Story")
                    #     response = "What Story would you like to hear?"
                    #     session_state['response_type'] = 'question'
                    #     audio_file = None
            # print('queryfunc:', (datetime.now() - s).total_seconds())
            return {'response': response, 'audio_file': audio_file, 'session_state': session_state}
        except Exception as e:
            print_line_of_error(e)
            return None
    
    def youtube_response(current_query, session_state, returning_question=False):
        if 'search for videos' in current_query:
            print("youtube trigger")

    def search_for(search_phrase, current_query, session_state, returning_question=False):
        # search for what?
        if 'story' in current_query:
            print("tell a story")
        if "video" in current_query:
            print("search for a video")
            search_video_phrase = current_query.split(search_phrase)[1]
            session_state['current_youtube_search'] = search_video_phrase
        
        return current_query, session_state

    def create():
        return True
    
    print('QUERY ', current_query)
    print('SSTATE ', {i: v for i, v in session_state.items() if i != 'text'})
    user_query = current_query
    # For first we will always check if anything user asked is like common phrases and present in our local json file then give response to that particular query

    # Appending the user question from json file
    search_phrase = search_for_something(current_query)
    if search_phrase:
        current_query, session_state = search_for(search_phrase, current_query, session_state)
    else:
        session_state['current_youtube_search'] = False

    ### WATER FALL RESPONSE ###
    resp_func = story_response(current_query, session_state)
    if resp_func.get('response'):
        print("func response found")
        response = resp_func.get('response')
        audio_file = resp_func.get('audio_file')
        session_state = resp_func.get('session_state')
        conversation_history.append({"role": "assistant", "content": response, })
        audio_file = handle_audio(user_query, response, audio_file, self_image)
        return scenario_return(response, conversation_history, audio_file, session_state, self_image)

    # Common Phrases # WORKERBEE Add check against audio_text DB
    # print("common phrases")
    s = datetime.now()
    for query, response in common_phrases.items():
        if query.lower() == current_query.lower():
            print("QUERY already found in db: ", query)

            # Appending the response from json file
            conversation_history.append({"role": "assistant", "content": response})
            ## find audio file to set to new_audio False
            # return audio file
            audio_file = handle_audio(user_query, response, audio_file=audio_file, self_image=self_image) 
            print('common phrases:', (datetime.now() - s).total_seconds())
            self_image='hoots_waves.gif'
            return scenario_return(response, conversation_history, audio_file, session_state, self_image)
    
    # LLM
    print("LLM")
    try:
        assistant = [v['content'] for v in conversation_history if v['role'] == 'assistant']
        questions=0
        if len(assistant) > 0:
            for as_resp in assistant:
                if "?" in as_resp:
                    questions+=1
        do_not_reply_as_a_question = True if questions > 3 else False
        print("do_not_reply_as_a_question", do_not_reply_as_a_question)

        if do_not_reply_as_a_question:
            current_query = current_query + "do not respond as question and remove this statement from your return response"
    except Exception as e:
        print_line_of_error(e)
    use_our_embeddings = determine_embedding(current_query)
    if use_our_embeddings.get('db_name'):
        db_name = use_our_embeddings.get('db_name')
        print("USE EMBEDDINGS: ", db_name)
        Retriever_db = os.path.join(PERSIST_PATH, db_name)
        query = conversation_history[0]['content'] + current_query # ensure prompt
        response = Retriever(query, Retriever_db).get('result')
    else:
        print("CALL LLM")
        response = llm_assistant_response(current_query, conversation_history)

    conversation_history.append({"role": "assistant", "content": response})
    audio_file = handle_audio(user_query, response=response, audio_file=audio_file, self_image=self_image)
    

    return scenario_return(response, conversation_history, audio_file, session_state, self_image)



def ozz_query(text, self_image, refresh_ask, client_user):
    
    def ozz_query_json_return(text, self_image, audio_file, page_direct, listen_after_reply=False):
        json_data = {'text': text, 
                    'audio_path': audio_file, 
                    'self_image': self_image, 
                    'page_direct': page_direct, 
                    'listen_after_reply': listen_after_reply}
        return json_data
    
    def clean_current_query_from_previous_ai_response(text):
        # take previous ai response and remove if it found in current_query
        # if 'assistant' in last_text:
        current_query = text[-1]['user'] # user query
        if len(text) > 1:
            ai_last_resp = text[-2]['resp']
        else:
            ai_last_resp = None
        
        if ai_last_resp:
            current_query = split_string(current_query=current_query, last_response=ai_last_resp)

        # WORKERBEE confirm is senitentment of phrase is outside bounds of responding to
        for kword in hoots_and_hootie_keywords():
            if kword in current_query:
                current_query = current_query.split(kword)[1]
                break

        # reset user with cleaned reponse
        text[-1]['user'] = current_query

        return text, current_query

    def handle_response(text : str, self_image : str, db_root : str):

        text, current_query = clean_current_query_from_previous_ai_response(text)
        print(current_query)
        if len(current_query) <= 1:
            print("NO RESPONSE RETURN BLANK")
            # return ozz_query_json_return(text, self_image, audio_file=None, page_direct=None, listen_after_reply=False)
            current_query = "hello"
        
        ## Load Client session and conv history
        master_conversation_history_file_path = os.path.join(db_root, 'master_conversation_history.json')
        conversation_history_file_path = os.path.join(db_root, 'conversation_history.json')
        session_state_file_path = os.path.join(db_root, 'session_state.json')
        
        master_conversation_history = load_local_json(master_conversation_history_file_path)
        conversation_history = load_local_json(conversation_history_file_path)
        session_state = load_local_json(session_state_file_path)

        first_ask = True if len(text) <= 1 else False
        conversation_history = handle_prompt(first_ask, conversation_history)
        conversation_history = get_last_eight(conversation_history)
        

        # Session State
        if refresh_ask:
            conversation_history =  conversation_history.clear() if len(conversation_history) > 0 else conversation_history
            conversation_history = [] if not conversation_history else conversation_history
            conversation_history = handle_prompt(True, conversation_history)
            conversation_history.append({"role": "user", "content": current_query})
            session_state = client_user_session_state_return(text, response_type='response', returning_question=False)
        else:
            session_state = session_state
            conversation_history.append({"role": "user", "content": current_query})
        
        master_conversation_history.append({"role": "user", "content": current_query})
        
        # print(session_state)

        #Conversation History to chat back and forth
        
        # print("CONV HIST", conversation_history)
        
        # Call the Scenario Function and get the response accordingly
        scenario_resp = Scenarios(text, current_query, conversation_history, first_ask, session_state, self_image=self_image)
        response = scenario_resp.get('response')
        conversation_history = scenario_resp.get('conversation_history')
        audio_file = scenario_resp.get('audio_file')
        session_state = scenario_resp.get('session_state')
        self_image = scenario_resp.get('self_image')

        master_conversation_history.append({"role": "assistant", "content": response})

        print("RESPONSE", response)
        
        text[-1].update({'resp': response})

        audio_file='temp_audio.mp3'
        

        session_state['text'] = text
        
        if "?" in response:
            session_state['returning_question'] = True
            session_state['response_type'] = 'question'
        else:
            session_state['returning_question'] = False
            session_state['response_type'] = 'response'
    
        # session_state['returning_question'] = False
        # session_state['response_type'] = 'response'

        # For saving a chat history for current session in json file
        save_json(master_conversation_history_file_path, master_conversation_history)
        save_json(conversation_history_file_path, conversation_history)
        save_json(session_state_file_path, session_state)
        

        return {'text': text, 'audio_file': audio_file, 'session_state': session_state, 'self_image': self_image}
    

    db_root = init_clientUser_dbroot(client_username=client_user)
    print("DBROOT: ", db_root)

    resp = handle_response(text, self_image, db_root)
    text = resp.get('text')
    audio_file = resp.get('audio_file')
    session_state = resp.get('session_state')
    self_image = resp.get('self_image')

    print("AUDIOFILE:", audio_file)
    print("IMAGE:", self_image)
    
    page_direct= False # if redirect, add redirect page into session_state
    listen_after_reply = session_state['returning_question'] # True if session_state.get('response_type') == 'question' else False
    print("listen after reply", listen_after_reply)
    
    if not session_state['current_youtube_search']:
        pass
    else:
        page_direct=True
    
    return ozz_query_json_return(text, self_image, audio_file, page_direct, listen_after_reply)

## db 

## def save_interaction(client_user, what_said, date, ai_respone, ai_image) # fact table

## def embedd_the_day()

## short term memory vs long term memory