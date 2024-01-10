

import base64
import openai
import os
from quart import send_file

from core.cartridges import addCartridge, update_cartridge_field
from file_handling.s3 import write_file
from file_handling.transcribe import transcribe_file
from tools.debug import eZprint



file_chunks = {}

from session.sessionHandler import novaSession

async def handle_file_start(data):
    eZprint('file start', ['FILE_HANDLING'], line_break=True)
    
    tempKey = data["tempKey"]
    file_chunks[tempKey] = {
    "metadata": data,   
    "chunks_received": 0,
    "content": [],
    }
    return True


async def handle_file_chunk(data):
    print('file chunk')
    tempKey = data["tempKey"]
    file_chunks[tempKey]["chunks_received"] += 1
    # Decode the base64-encoded chunkContent
    decoded_chunk_content = base64.b64decode(data["chunkContent"])
    file_chunks[tempKey]["content"].append(decoded_chunk_content)
    return file_chunks[tempKey]["chunks_received"]
  # You could also process and store the chunk immediately in this step
  # instead of collecting all chunks in `file_chunks` and processing them later

async def handle_file_end(data):
    print('file end')
    tempKey = data["tempKey"]
    file_metadata = file_chunks[tempKey]["metadata"]
    file_content = b''.join(file_chunks[tempKey]["content"])
    # Process the uploaded file
    # You might need to convert the content from a bytearray to the initial format (e.g., base64)
    # print(file_metadata)
    data = {
        'sessionID': file_metadata['sessionID'],
        'userID': file_metadata['userID'],
        'file_content': file_content,
        'file_name': file_metadata['file_name'],
        'file_type': file_metadata['file_type'],
        'sessionID': file_metadata['sessionID'],
        'convoID' : file_metadata['convoID'],
        'loadout': file_metadata['loadout'],
        'tempKey': file_metadata['tempKey'],
        'document_type': file_metadata['document_type'],
    }

    sessionID = data['sessionID']
    # client_loadout = None
    # if sessionID in current_loadout:
    #     client_loadout = current_loadout[sessionID]
        
    file_content = data['file_content']

    file_name = data['file_name']
    file_type = data['file_type']
    loadout = data['loadout']
    convoID = data['convoID']

    convoID = novaSession[sessionID]['convoID']
    cartVal = {
        'label' : file_name,
        # 'text' : str(transcriptions),
        'file' : file_name,
        'extension' : file_type,
        # 'media_url' : url,
        'type' : 'media',
        'enabled' : True,
    }

    extension = file_type.split('/')[1]
    if extension == 'quicktime':
        extension = 'mov'
    if extension == 'x-matroska':
        extension = 'mkv'
    if extension == 'mpeg':
        extension = 'mp3'
    if extension == 'plain':
        extension = 'txt'


    cartKey = await addCartridge(cartVal, sessionID, loadout, convoID)
    file_name_to_write = cartKey + '.' + extension

    transcript_text = await transcribe_file(file_content, cartKey, file_name, file_type, sessionID, convoID, loadout)
 
    url = await write_file(file_content, file_name_to_write) 

    eZprint(f'file {file_name_to_write} written to {url}', ['FILE_HANDLING'])

    await update_cartridge_field({'sessionID': sessionID, 'cartKey' : cartKey, 'fields': {
        'media_url': url,
        'aws_key': file_name_to_write
        }}, convoID, loadout, True)
    
    del file_chunks[tempKey]
    return file_name + ' recieved' + ' ' + str(transcript_text)

async def get_file_download_link(filename):    
    return await send_file(filename, attachment_filename=filename, as_attachment=True)

