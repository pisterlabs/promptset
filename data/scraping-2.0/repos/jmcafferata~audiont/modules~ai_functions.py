import modules.clean_options as clean
import openai
import modules.convert_to_wav as convert
import modules.csv_manipulation as csvm
import modules.decode_utf8 as decode
import json
from pathlib import Path # library used to handle file paths // librería usada para manejar rutas de archivos
import config as config # import the config file // importar el archivo de configuración
import pandas as pd # library used to handle dataframes // librería usada para manejar dataframes
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import numpy as np
from ast import literal_eval
from datetime import datetime
import pytz
from io import StringIO
import os
import requests
from bs4 import BeautifulSoup
import sys
import traceback
import re
import csv
from nanoid import generate
import asyncio
from telegram.error import BadRequest
from telegram.error import RetryAfter


# set the OpenAI API key, so the code can access the API // establecer la clave de la API de OpenAI, para que el código pueda acceder a la API
openai.api_key = config.openai_api_key
timezone = pytz.timezone('America/Argentina/Buenos_Aires')

def ensure_csv_extension(database_name):
    file_name, file_extension = os.path.splitext(database_name)
    if file_extension.lower() != '.csv':
        file_name = database_name + '.csv'  # Append '.csv' to the original database_name
    else:
        file_name = database_name  # If the extension is already '.csv', use the original database_name
    return file_name

# read key-value pairs from csv file // leer pares clave-valor de un archivo csv
def read_data_from_csv(key: str, filename: str) -> dict:
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if row[0].strip() == key.strip():
                # return the value of the key // devolver el valor de la clave
                return row[1]
    # return None if the key is not found // devolver None si la clave no se encuentra
    return None



def check_and_compute_cosine_similarity(x, message_vector):
    x = np.array(literal_eval(x), dtype=np.float64)  # Convert x to float64
    return cosine_similarity(x, message_vector)

################## GOOGLE SEARCH ############################

def get_chunks(url,text):
   
    # get the html from the url
    response = requests.get(url)
    print("########### ENTERING GET CHUNKS ###########")
    html = response.text

    # parse the html
    soup = BeautifulSoup(html, "html.parser")

    tag_types = ["p", "ul", "h1", "h2", "blockquote"]

    # create an empty dataframe
    df = pd.DataFrame(columns=["text", "embedding"])

    
    for tag_type in tag_types:
        tags = soup.find_all(tag_type)
        for tag in tags:
            # check tag isnt empty
            if tag.text == "":
                continue
            tag_embedding = get_embedding(str(tag.text), "text-embedding-ada-002")
            print(tag.text)

            # Create a new DataFrame for the current row
            new_row = pd.DataFrame({"text": [tag.text], "embedding": [tag_embedding]})

            # Concatenate the existing DataFrame (df) with the new_row DataFrame
            df = pd.concat([df, new_row], ignore_index=True)
        
    
    # get the embeddings for the query
    query_embedding = get_embedding(text,'text-embedding-ada-002')

    # calculate the cosine similarity between the query and each chunk
    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(x, query_embedding))

    # sort the chunks by similarity
    df = df.sort_values(by="similarity", ascending=False)

    # get the top 5 chunks
    top_chunks = df["text"].head(1).values

    # join the top 5 chunks into a single string
    summary_chunks = " ".join(top_chunks)

    print(summary_chunks)

    # print the summary

    return summary_chunks
 
def google(query,message):
    # delete anything that comes after a \n
    query = re.sub(r"\\n.*", "", query)
    # delete anything that comes after a -
    query = re.sub(r"-.*", "", query)

    print("Query:", query)

    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": config.google_api_key,
        "cx": '35af1fa5e6c614873',
        "q": query,
    }

    
    max_attempts = 3
    success = False

    for attempt in range(max_attempts):
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'items' in data:
                success = True
                break
            else:
                print(f"No results found or an error occurred on attempt {attempt + 1}")
                print("Response data:", data)

        except requests.exceptions.HTTPError as e:
            print(f"An error occurred on attempt {attempt + 1}: {e}")

        if not success:
            print("Trying again...")

    if not success:
        print("All attempts failed.")
        return
    
        
    # get the link and snippet elements from the first 10 items
    chunks_and_links = []

    max_items = len(data["items"])
    # print all links
    item_index = 0

    while item_index < max_items:
        item = data["items"][item_index]
        url = item["link"]
        
        print("######## LINK ########")

        # Verificar si la URL no es un archivo PDF
        if url.lower().endswith(".pdf"):
            item_index += 1
            continue

        print(url)
        result_chunks = get_chunks(url, message)
        chunks_and_links.append({
            "link": url,
            "snippet": item["snippet"],
            "chunks": result_chunks
        })

    item_index += 1

    
    return chunks_and_links

async def find_in_google(message):
    
    print("About to ask GPT-3 for a query to search in Google for the following task:\n"+message)
    # get latest message from the user
    gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":"Create the perfect google query to get information for the following task:\n"+message}]
        )
    gpt_response = gpt_response.choices[0].message.content
    print("################ GPT-3 response ############",gpt_response)
    querychunks_and_links = google(gpt_response,message)
    results_chunks = ""
    for chunk in querychunks_and_links:
        results_chunks += chunk["chunks"]+"\n"
    return results_chunks

################### AUDIO TRANSCRIPTION ############################

# use the OpenAI API to get the text from the audio file // usar la API de OpenAI para obtener el texto del archivo de audio 
async def transcribe_audio(update):
    # Extract the audio file from the message // Extraer el archivo de audio del mensaje
    new_file = await update.message.effective_attachment.get_file()
    # Download the audio file to the computer // Descargar el archivo de audio al computador
    await new_file.download_to_drive()
    # get the name of the audio file // obtener el nombre del archivo de audio
    file_path = Path(new_file.file_path).name
    # convert the audio file to a wav file // convertir el archivo de audio a un archivo wav
    try:
        new_audio_path = convert.convert_to_wav(file_path)
    except Exception as e:
        print("⬇️⬇️⬇️⬇️ Error en la conversión de audio ⬇️⬇️⬇️⬇️\n",e)
        # send a message to the user, telling them there was an error // enviar un mensaje al usuario, diciéndoles que hubo un error
        await update.message.reply_text('Hubo un error en la conversión de audio. Probá de nuevo.')
        # send e
        await update.message.reply_text(str(e))
        # send an images/cat.jpg to the user // enviar una imagen a la usuaria
        await update.message.reply_photo(open('images/cat.jpg', 'rb'))
    else:
        # open the wav file // abrir el archivo wav
        wav_audio = open(new_audio_path, "rb")
        # call the OpenAI API to get the text from the audio file // llamar a la API de OpenAI para obtener el texto del archivo de audio
        try:
            print("⬆️⬆️⬆️⬆️ Starting transcription ⬆️⬆️⬆️⬆️")
            transcription_object = openai.Audio.transcribe(
            "whisper-1", wav_audio, language="es", prompt="esto es una nota de voz. hay momentos de silencio en el audio, cuidado con eso.")
            print("Transcription:\n"+transcription_object["text"])
        except Exception as e:
            print("⬇️⬇️⬇️⬇️ Error en la transcripción de audio ⬇️⬇️⬇️⬇️\n",e)
            # send a message to the user, telling them there was an error // enviar un mensaje al usuario, diciéndoles que hubo un error
            await update.message.reply_text('Hubo un error en la transcripción de audio. Probá de nuevo.')
            # send e
            await update.message.reply_text(str(e))
            # send an images/cat.jpg to the user // enviar una imagen a la usuaria
            await update.message.reply_photo(open('images/sad.jpg', 'rb'))
            raise e
        else:
            print("Transcription:\n"+transcription_object["text"])
            # delete audio files from the server
            wav_audio.close()
            os.remove(file_path)
            os.remove(new_audio_path)


            return transcription_object["text"]


################### v3 ############################

def get_top_entries(db, query, top_n=15):

    with open(db, 'rb') as f:
        csv_str = f.read()
    entries_df = pd.read_csv(StringIO(csv_str.decode('utf-8')), sep='|', encoding='utf-8', escapechar='\\')
    query_vector = get_embedding(query, 'text-embedding-ada-002')

    entries_df['similarity'] = entries_df['embedding'].apply(lambda x: check_and_compute_cosine_similarity(x, query_vector))

    # sort by similarity
    entries_df = entries_df.sort_values(by=['similarity'], ascending=False)

    # Initialize a string with the df headers
    headers = list(entries_df.columns)
    headers.remove('similarity')
    similar_entries = ' | '.join(headers) + '\n'

    # Iterate over the rows of the DataFrame
    for index, row in entries_df.head(top_n).iterrows():
        # Replace the value of the 'embedding'  with ...
        row['embedding'] = '...'
        # delete the similarity column
        del row['similarity']
        # Convert the row to a string and join the values using '|'
        row_str = ' | '.join(map(str, row.values))
        # Append the row string to 'similar_entries' followed by a newline character
        similar_entries += row_str + '\n'

    return similar_entries


async def chat(update,message,model,personality):
    now = datetime.now()
    print("################# ENTERING CHAT MODE #################")

    # get 10 most recent messages from chat.csv
    chat_df = pd.read_csv('chat.csv', sep='|', encoding='utf-8', escapechar='\\')
    chat_df = chat_df.tail(10)
    # for each message, get the role and content
    prompt = []
    prompt.append({"role": "system", "content": "Today is " + now.strftime("%d/%m/%Y %H:%M:%S")+ "\n"+config.personalidad})
    for index, row in chat_df.iterrows():
        prompt.append({"role": row['role'], "content": row['content']})
    prompt.append({"role": "user", "content": message})
    if model == "3":
        model = "gpt-3.5-turbo"
    else:
        model = "gpt-4"
    gpt_response = openai.ChatCompletion.create(
        model=model,
        messages=prompt,
    )
    print("################ CHAT response ############", gpt_response.choices[0].message.content)
    #store the message on chat.csv. make sure you replace the new lines with spaces
    with open('chat.csv', 'a', encoding='utf-8') as f:
        f.write(now.strftime("%d/%m/%Y %H:%M:%S")+'|user|'+message.replace('\n', ' ')+'\n')
    # store the response on chat.csv but replace new lines with \n
    with open('chat.csv', 'a', encoding='utf-8') as f:
        f.write(now.strftime("%d/%m/%Y %H:%M:%S")+'|assistant|' + gpt_response.choices[0].message.content.replace('\n', ' ') + '\n')
    
    return gpt_response.choices[0].message.content

async def secretary(update,message,personality,context):
    now = datetime.now()

    #check if there is a username and a full name
    if update.message.from_user.username is None:
        #get user id
        username = "User ID: "+update.message.from_user.id
    else:
        #get user username
        username = update.message.from_user.username
    #check if full name is none
    if update.message.from_user.full_name is None:
        #get user id
        full_name = "User ID: "+update.message.from_user.id
    else:
        full_name = update.message.from_user.full_name
    
    #embed and store the message in db/messages.csv
    message_vector = get_embedding(message, 'text-embedding-ada-002')

    with open('db/messages.csv', 'a', encoding='utf-8') as f:
        f.write(now.strftime("%d/%m/%Y %H:%M:%S")+'|'+full_name+'|'+message.replace('\n', ' ')+'|'+str(message_vector)+'|'+str(generate('0123456789', 5))+'\n')
    
    #get top entries from db/notes.csv
    related_notes = get_top_entries('db/notes.csv', message, 10)
    #truncate the string to 2000 characters
    related_notes = related_notes[:2000]
    print("related_notes", related_notes)
    prompt = []

    prompt.append({"role": "system", "content": "Today is " + now.strftime("%d/%m/%Y %H:%M:%S")+ "\n"+config.personalidad+"\nMantené tus respuestas a menos de 100 caracteres.\nAcá van algunas notas de "+config.my_name+" que pueden ayudar:\n"+related_notes+" \nMi nombre es "+full_name+" ("+username+")"})
    #print all those values to check their type
    prompt.append({"role": "user", "content": message})
    gpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=prompt,
        stream=True
    )
    collected_chunks = []
    collected_messages = []
    # iterate through the stream of events
    new_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="✍️✍️✍️✍️")
    current_text = ''
    for chunk in gpt_response:
        collected_chunks.append(chunk)  # save the event response
        chunk_dict = chunk['choices'][0]['delta']  # extract the message
        chunk_message = chunk_dict.get('content')  # extract the message text

        if chunk_message is not None:
            collected_messages.append(chunk_message)  # save the message
            print("chunk_message: ", chunk_message)
            legible_text = ''.join(collected_messages)
            if legible_text != current_text:
                # Edit the message with the concatenated response text
                try:
                    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=new_message.message_id, text=legible_text)
                    current_text = legible_text
                    # await asyncio.sleep(0.1)  # Add a small delay before editing the message again
                except BadRequest as e:
                    if 'Message is not modified' not in str(e):
                        raise e
    


async def crud(update,message,context):

    final_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="🤖🤖🤖🤖")

    now = datetime.now()
    
    with open('chat.csv', 'a', encoding='utf-8') as f:
        f.write(now.strftime("%d/%m/%Y %H:%M:%S") + "|user|" + message.replace("\n", " ") + "\n")

    # get 'step-one' value from instructions.csv
    step_one_instructions = read_data_from_csv('crud', 'instructions.csv')
    prompt = []
    # get a list of all the csv files in the folder
    csv_files = [f for f in os.listdir('./db/') if os.path.isfile(os.path.join('./db/', f)) and f.endswith('.csv')]
    print("############## ENTERING STEP ONE #################")
    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=final_message.message_id, text="Entrando al paso 1...")
    print("csv_files: " + str(csv_files))
    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=final_message.message_id, text="csv_files: " + str(csv_files))
    headers_and_first_rows = ''
    #for each csv
    for csv_file in csv_files:
        #open it
        with open(os.path.join('./db/', csv_file), 'r') as f:
            csv_reader = csv.reader(f, delimiter='|')
            print("csv_file: " + csv_file)
            # get the first two lines
            header = next(csv_reader)
            try:
                first_row = next(csv_reader) 
            except StopIteration:
                first_row = []
            excluded_column_name = "embedding" # replace with the name of the column you want to exclude
            excluded_column_index = header.index(excluded_column_name)
            
            if first_row != []:
                first_row[excluded_column_index] = "..."
            
            # combine the remaining columns into a string
            header_string = '|'.join(header)
            first_row_string = '|'.join(first_row)
            # add the strings to the list
            headers_and_first_rows += csv_file+': \n'+header_string +"\n"+ first_row_string
   
    prompt.append({"role": "system", "content": "Today is " + now.strftime("%d/%m/%Y %H:%M:%S") + ".\n" + step_one_instructions})
    prompt.append({"role": "user", "content": message})
    prompt.append({"role": "user", "content": "These are the only available database files to perform CRUD (with headers and first row): " +headers_and_first_rows})
    # esto devuelve un JSON o un list de JSONs
    step_one_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        temperature=0.2,
    )

    response_string = step_one_response.choices[0].message.content
    print('step one response_string \n', response_string)
    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=final_message.message_id, text="Paso 1: \n"+response_string)
    # if response contains '[', get everything between the first '[' and the last ']'
    print("############## ENTERING STEP TWO ###############")
    
    if '[' in response_string:
        databases_to_crud_string = response_string[response_string.find('['):response_string.rfind(']')+1]
        databases_to_crud = json.loads(databases_to_crud_string)
        print("databases_to_crud: " + str(databases_to_crud))

        database_info = ''

        # for each database in the list
        for database in databases_to_crud:
            # get the database name
            database_name = os.path.join('./db/', ensure_csv_extension(database["database"]))
            database_prompt = database['prompt']
            database_info += 'Database name: '+database["database"] + '. Prompt: ' + database_prompt + '\n'
            # open the database csv file
            with open(database_name, 'r', encoding='utf-8') as f:
                # read the entire file into lines
                lines = f.readlines()
                
            # get the header
            header = lines[0].strip()
            database_info += 'Header: ' + header + '\n'
            
             # get the database similar rows
            print("database_name: ",database_name)
            similar_entries = get_top_entries(database_name, database_prompt, 7)
            #truncate the string to 2300 characters
            similar_entries = similar_entries[:1500]
            database_info += similar_entries + '\n'
        print("database_info: \n",database_info)
        info_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Info que me va a ayudar: \n"+database_info)

        step_two_instructions = read_data_from_csv('step-two', 'instructions.csv')
        prompt = []
        prompt.append({"role": "system", "content":"Today is " + now.strftime("%d/%m/%Y %H:%M:%S") + ".\n" + step_two_instructions})
        prompt.append({"role": "user", "content": message})
        prompt.append({"role": "user", "content": "Use this database to write the response:\n" + database_info})
        step_two_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0.8,
        )

        step_two_response_string = step_two_response.choices[0].message.content
        print('step two response_string \n', step_two_response_string)
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=final_message.message_id, text="Paso 2: \n"+step_two_response_string)
        print("############## ENTERING STEP THREE ###############")
        # if response contains '[', get everything between the first '[' and the last ']'
        if '[' in step_two_response_string:
            database_commands_string = step_two_response_string[step_two_response_string.find('['):step_two_response_string.rfind(']')+1]
            database_commands = json.loads(database_commands_string)
            for command in database_commands:
                if command["operation"] == "create":
                    print("creating: " + str(command))
                    csv_path = os.path.join('./db/', ensure_csv_extension(command['database']))
                    
                    # Read the CSV file into a DataFrame
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        csv_str = f.read()
                    df = pd.read_csv(StringIO(csv_str), sep='|', encoding='utf-8', escapechar='\\')
                    
                    

                    # Check if all command fields are in the DataFrame columns
                    missing_columns = set(command['row'].keys()) - set(df.columns)
                    
                    # Add missing columns and fill the old rows with empty strings
                    for col in missing_columns:
                        df[col] = ''
                    
                    # Move the 'embedding' column to the end
                    cols = [col for col in df.columns if col != 'embedding'] + ['embedding']
                    df = df[cols]
                    
                    # Append the new row
                    row_data = command['row']
                    row_data['row_id'] = generate('0123456789', 5)
                    row_data_string = str(row_data)
                    row_data["embedding"] = get_embedding(row_data_string, "text-embedding-ada-002")
                    
                    row_to_add = pd.DataFrame([row_data], columns=df.columns)
                    df = pd.concat([df, row_to_add], ignore_index=True)  # Use pandas.concat instead of frame.append
                    
                    # Write the updated DataFrame back to the CSV file
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        df.to_csv(f, sep='|', index=False, escapechar='\\', encoding='utf-8')

                    
                    return "Added:\n" + row_data_string + "\n to database " + command["database"]
                elif command["operation"] == "read":
                    print("reading: " + str(command))
                    similarity_results = get_top_entries(os.path.join('./db/', ensure_csv_extension(command['database'])), command['prompt'], 7)
                    #send the response but make the string max 4000 characters
                    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=final_message.message_id, text="Paso 3: \n" + similarity_results[:3000])
                    now = datetime.now()
                    prompt = []
                    prompt.append({"role": "system", "content": config.personalidad + "\nHoy es " + now.strftime("%d/%m/%Y %H:%M:%S") + ".\n"})
                    prompt.append({"role": "user", "content": message})
                    prompt.append({"role": "user", "content": "Use this data to write the response with UTF-8 encoding:\n" + similarity_results})
                    #writ the similarity results to the chat.csv file
                    with open('chat.csv', 'a', encoding='utf-8') as f:
                        now = datetime.now()
                        f.write(now.strftime("%d/%m/%Y %H:%M:%S") + "|user|Acá te paso data mía:" + similarity_results.replace("\n", " ").replace("|", " ") + " - ")

                    step_three_response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=prompt,
                        temperature=0.8,
                        stream=True
                    )
                    # create variables to collect the stream of chunks


                    collected_chunks = []
                    collected_messages = []
                    # iterate through the stream of events
                    new_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Respuesta:\n")
                    current_text = ''
                    for chunk in step_three_response:
                        collected_chunks.append(chunk)  # save the event response
                        chunk_dict = chunk['choices'][0]['delta']  # extract the message
                        chunk_message = chunk_dict.get('content')  # extract the message text

                        if chunk_message is not None:
                            collected_messages.append(chunk_message)  # save the message
                            print("chunk_message: ", chunk_message)
                            legible_text = ''.join(collected_messages)
                            if legible_text != current_text:
                                # Edit the message with the concatenated response text
                                try:
                                    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=new_message.message_id, text=legible_text)
                                    current_text = legible_text
                      
                                    # await asyncio.sleep(0.1)  # Add a small delay before editing the message again
                                except BadRequest as e:
                                    if 'Message is not modified' not in str(e):
                                        raise e
                                except RetryAfter as e:
                                    counter = 0
                                    counter_max = e.retry_after
                                    error_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Me estoy quedando sin memoria, por favor, espera un momento... ( " + str(e.retry_after) + " segundos)")
                                    #edit the error message like a timer
                                    while counter < counter_max:
                                        edited_error_message = "Me estoy quedando sin memoria, por favor, espera un momento... ( " + str(counter_max - counter) + " segundos)"
                                        await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=error_message.message_id, text=edited_error_message)
                                        await asyncio.sleep(1)
                                        counter += 1
                                        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=error_message.message_id)

                                    current_text = legible_text
                                    if current_text != '':
                                        # delete the final_message
                                        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=final_message.message_id)
                                            
                                        with open('chat.csv', 'a', encoding='utf-8') as f:
                                            now = datetime.now()
                                            f.write(now.strftime("%d/%m/%Y %H:%M:%S") + "|assistant|" + current_text.replace("\n", " ").replace("|", " ") + "\n")
                                    await asyncio.sleep(e.retry_after)
                                    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=new_message.message_id, text=legible_text)

                                    # await asyncio.sleep(0.1)  # Add a small delay before editing the message again
                    # write the user message and the legible text to chat.csv (date|role|content)
                    if current_text != '':
                        # delete the final_message
                        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=final_message.message_id)
                        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=info_message.message_id)
                        print("current_text: ", current_text)
                        print("writing to chat.csv")
                        with open('chat.csv', 'a', encoding='utf-8') as f:
                            now = datetime.now()
                            f.write(now.strftime("%d/%m/%Y %H:%M:%S") + "|assistant|" + current_text.replace("\n", " ").replace("|", " ") + "\n")
            
                    

                elif command["operation"] == "update":
                    # todo: update an entry
                    print("updating: " + str(command))
                    return str(database_commands)
                elif command["operation"] == "delete":
                    # todo: delete an entry
                    print("deleting: " + str(command))
                    return str(database_commands) 
        else:
            return step_two_response.choices[0].message.content
    else:
        return step_one_response.choices[0].message.content
    
   





    step_two_instructions = read_data_from_csv('step-two', 'instructions.csv')
    prompt = []
    prompt.append({"role": "system", "content": step_two_instructions})
    prompt.append({"role": "user", "content": update.message.text})


    return str(similar_entries)


################### v1 ############################

async def complete_prompt(reason, message,username,update,personality):

    user_name = 'Anónimo'
    # THE JUICE // EL JUGO

    now = datetime.now(timezone)

    chat_messages = []
    chat_messages.append({"role":"system","content":config.personalidad + "\n Today is " + now.strftime("%d/%m/%Y %H:%M:%S")})
    

    if (reason == "summary"):
        model = "gpt-3.5-turbo"
        clean.options = []
        chat_messages.append({"role":"user","content":"""
        Hola, me acaban de enviar un mensaje de voz. Dice lo siguiente:
        
        """ + message + """

        Tengo que responder este mensaje. Haceme un resumen del mensaje y dame 4 opciones de respuesta:
        las primeras 3 son contextuales y son títulos de posibles respuestas distintas (positiva, negativa, neutral) que se le podría dar al mensaje de voz (máximo 40 caracteres). la cuarta opción es -Otra respuesta-.
        Las opciones tienen que estar al final y con el formato 1.,2.,3.,4. y deben contener emojis que ilustren el sentimiento de la respuesta."""})

        
    elif (reason == "answer"):
        # get related notes from db/notes.csv using pandas cosine similiaryty and get_top_entries
        notes_file = 'db/notes.csv'
        # get the top 3 entries
        similar_entries = get_top_entries(notes_file,csvm.get_last_audio(), 7)


        model = "gpt-4"
        chat_messages.append({"role":"user","content":"""
        Me acaban de enviar un mensaje de voz. Dice lo siguiente:
    
        """ + csvm.get_last_audio() + """
        
        Tengo que responder este mensaje con las siguientes instrucciones:
        
        """ + message + """
        
        Escribir el mensaje de parte mía hacia el remitente, usando mis instrucciones como guía (pero no es necesario poner literalmente lo que dice la instrucción), mi personalidad y usando el mismo tono conversacional que mi interlocutor. Usar las siguientes notas mías para escribir igual a mí.
         
          """ + similar_entries + """
        Que sea muy natural, que parezca el cuerpo de un mensaje de chat."""})

        
    

    elif (reason == "assistance"):
        chat_messages.append({"role":"user","content":message})



    elif (reason == "google"):

        
        #internet_information tiene que ser un string
        internet_information = await find_in_google(message)
        
        chat_messages.append({"role":"user","content":
        message + 
        "\nUsá la siguiente información para responder el mensaje:\n\n"
        })
        chat_messages.append({"role":"user","content":internet_information})



    print("############### FIN DE CONFIGURACIÓN DEL PROMPT ################")
    print("############### PROMPTING THE AI ################")
    print('chat_messages: ', chat_messages)

    gpt_response = openai.ChatCompletion.create(
    model=model,
    messages=chat_messages,
    )

    
    # get the text from the response // obtener el texto de la respuesta
    text = (gpt_response.choices[0].message.content)
    # decode the text // decodificar el texto
    decoded_text = decode.decode_utf8(text)

    # decoded text without new lines
    decoded_text_without_new_lines = decoded_text.replace("\n"," - ")

    #store the message in the csv // almacenar el mensaje en el csv
    # if the message doesn't start with the word GOOGLE (indicating that it's not a google search), store it in the csv // si el mensaje no comienza con la palabra GOOGLE (indicando que no es una búsqueda de Google), guárdelo en el csv
    print("Response:\n"+decoded_text)
    return decoded_text

        

    

################## EMBEDDINGS ##################

# use the Embeddings API to turn text into vectors // usar la API de Embeddings para convertir texto en vectores
def generate_embeddings(file,column_to_embed):
    # read the csv file // leer el archivo csv
    df = pd.read_csv(file)
    # create a new column to store the embeddings // crear una nueva columna para almacenar los incrustaciones
    df["embeddings"] = ""
    # loop through the rows of the dataframe // bucle a través de las filas del dataframe
    for index, row in df.iterrows():
        # get the text from the row // obtener el texto de la fila
        text = row[column_to_embed]
        # get the embedding from the text // obtener el incrustación del texto
        embedding = get_embedding(text)
        # store the embedding in the dataframe // almacenar el incrustación en el dataframe
        df.at[index, "embeddings"] = embedding
    # save the dataframe to a csv file // guardar el dataframe en un archivo csv
    df.to_csv(file, index=False)