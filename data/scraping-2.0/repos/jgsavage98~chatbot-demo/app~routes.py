from flask import Blueprint, request, jsonify, render_template, Response
import requests
import openai
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/create_meditation', methods=['POST'])
def create_meditation():
    data = request.json
    mood = data.get('mood')
    # music = data.get('music')
    goal = data.get('goal')
    # duration = data.get('duration')

    try:
        # Generate meditation script using OpenAI
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Create a guided meditation script. Mood: {mood}, Goal: {goal}.",
            max_tokens=1000
        )
        script = response.choices[0].text.strip()
        logger.info('********* Start of Script ********\n')
        logger.info(script)
        logger.info('********* End of Script ********\n')

        
        # Chunk the script
        chunks = chunk_script(script)
        
        # Azure TTS details
        azure_endpoint = "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1"
        azure_key = os.getenv('AZURE_TTS_KEY')
        headers = {
            'Ocp-Apim-Subscription-Key': azure_key,
            'Content-Type': 'application/ssml+xml',
            'X-Microsoft-OutputFormat': 'audio-16khz-128kbitrate-mono-mp3'
        }

        # Process each chunk
        combined_audio = bytearray()
        print("Number of chunks: ", len(chunks))
        counter = 1
        for chunk in chunks:
            print("Chunk Number ", counter)
            ssml = f"<speak version='1.0' xml:lang='en-US'> <voice xml:lang='en-US' xml:gender='Female' name='en-US-JennyNeural'>{chunk}</voice> </speak>"
            logger.info('********* Start of SSML Chunk ********\n')
            logger.info(ssml)
            logger.info('********* End of SSML Chunk ********\n')

            response = requests.post(azure_endpoint, headers=headers, data=ssml.encode('utf-8'))
            if response.status_code == 200:
                combined_audio.extend(response.content)
            else:
                raise Exception(f"Error from Azure TTS service: {response.status_code}")
            counter+=1

        return Response(bytes(combined_audio), mimetype='audio/mpeg')

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

def chunk_script(script, chunk_length=1000):
    
    #Divide the script into smaller parts. Each part should be small enough for Azure TTS to handle.
    
    chunks = []
    while script:
        chunk = script[:chunk_length]
        script = script[chunk_length:]
        chunks.append(chunk)
    return chunks

#if __name__ == '__main__':
    # Run the app



    #except Exception as e:
    #    return jsonify({"error": str(e)}), 500
