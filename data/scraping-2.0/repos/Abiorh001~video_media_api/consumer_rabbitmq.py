import pika
import os
import requests
import openai
import json

openai.api_key = "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"

def update_video_with_transcript(transcript, video_id):
    url = f'https://malzahra.tech/api/videos/{video_id}/'
    data = {
        'transcript': transcript,
    }

    try:
        response = requests.put(url, data={'transcript': transcript})
        if response.status_code == 200:
            print(f"Transcript for video {video_id} updated successfully.")
        else:
            print(f"Error updating transcript for video {video_id}. Status code:", response.status_code)

    except Exception as e:
        print(f"Error updating transcript for video {video_id}:", str(e))

def process_audio_file_path(ch, method, properties, body):
    try:
        message = json.loads(body)
        audio_file_path = message.get('audio_file_path')
        video_id = message.get('video_id')
        

        if not audio_file_path or not video_id:
            print("Invalid message format. Missing audio_file_path, video_id.")
            return
        
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.translate("whisper-1", audio_file)
            text = transcript.get('text')
            print(text)
           
            update_video_with_transcript(text, video_id)

    except Exception as e:
        print("Error processing audio:", str(e))

    ch.basic_ack(delivery_tag=method.delivery_tag)

def consume_audio_paths_from_rabbitmq():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='audio_file_paths')
    channel.basic_consume(queue='audio_file_paths', on_message_callback=process_audio_file_path)

    print("Waiting for audio file paths. To exit, press CTRL+C")
    channel.start_consuming()

if __name__ == "__main__":
    consume_audio_paths_from_rabbitmq()
