import argparse
import openai
import threading
import queue
import requests
import os
import nltk

# Initialize OpenAI API key
openai.api_key = ''
coqui_key = ''

voice_id = None

# Thread-safe queue for sentences
sentence_queue = queue.Queue()

# Priority queue for audio data and associated condition variable
audio_priority_queue = queue.PriorityQueue()
audio_data_condition = threading.Condition()

# FIFO file path
FIFO_PATH = "/tmp/audio_fifo"
# FIFO_PATH = "audio_fifo.out"

# Create FIFO file if it doesn't exist
# if False:
if not os.path.exists(FIFO_PATH):
    os.mkfifo(FIFO_PATH)

def call_tts():
    while True:
        order, sentence = sentence_queue.get()
        print(f"{order} Processing sentence: {sentence}")
        try:
            # Make a web request to the TTS service
            request_obj = {
                "voice_id": voice_id,
                "text": sentence

            }
            headers = {
                "accept": "audio/wav",
                "content-type": "application/json",
                "authorization": f"Bearer {coqui_key}"
            }
            response = requests.post("https://app.coqui.ai/api/v2/samples", json=request_obj, headers=headers)
            
            # Ensure the request was successful
            if response.status_code == 200:
                audio_data = response.content
                with audio_data_condition:
                    audio_priority_queue.put((order, audio_data))
                    audio_data_condition.notify_all()
            else:
                # Log the error for diagnosis
                print(f"Error {response.status_code} from TTS service: {response.text}")
                audio_priority_queue.put((order, None))
        except requests.RequestException as e:
            # Handle any exceptions raised by the requests library
            print(f"Error calling TTS service: {e}")
            audio_priority_queue.put((order, None))
        finally:
            sentence_queue.task_done()

def write_audio_to_fifo():
    current_order = 0
    with open(FIFO_PATH, 'wb') as fifo:
        while True:
            with audio_data_condition:
                while not audio_priority_queue.queue or audio_priority_queue.queue[0][0] != current_order:
                    audio_data_condition.wait()
                order, audio_data = audio_priority_queue.get()
                if(audio_data):
                    fifo.write(audio_data)
                audio_priority_queue.task_done()
                print(f"Audio Data {order} has been processed.")
                current_order += 1

def extract_full_sentence(overall_result):
    # Use NLTK to tokenize the text into sentences
    sentences = nltk.sent_tokenize(overall_result)
    if sentences and len(sentences) > 1:
        return sentences[0]
    return None

def main():
    global voice_id

    # Arguments
    parser = argparse.ArgumentParser("Quickly pipe the GPT output to audio file")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice_id", default='774437df-2959-4a01-8a44-a93097f8e8d5')
    parser.add_argument("--assistant_prompt", default="You are a helpful assistant.")
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--max_tokens", default=250)
    args = parser.parse_args()
    voice_id = args.voice_id


    overall_result = ""
    full_result = ""
    sentence_order = 0

    # Start TTS worker threads
    num_worker_threads = 5
    for _ in range(num_worker_threads):
        threading.Thread(target=call_tts, daemon=True).start()

    # Start FIFO writer thread
    threading.Thread(target=write_audio_to_fifo, daemon=True).start()

    # Connect to ChatGPT API in streaming mode
    response = openai.ChatCompletion.create(
        model=args.model,
        messages=[
      {
        "role": "system",
        "content": args.assistant_prompt
      },
      {
        "role": "user",
        "content": args.text
      }
    ],
        max_tokens=args.max_tokens,
        stream=True
    )

    for message in response:
        try:
            chunk = message['choices'][0]['delta']["content"]
        except:
            chunk = ""
            pass

        overall_result += chunk
        full_result += chunk

        sentence = extract_full_sentence(overall_result)
        while sentence:
            overall_result = overall_result[len(sentence):]
            sentence_queue.put((sentence_order, sentence))
            sentence_order += 1
            sentence = extract_full_sentence(overall_result)
    if(overall_result != None and overall_result != ""):
        sentence_queue.put((sentence_order, overall_result))
        sentence_order += 1
    # Wait for all tasks in the queue to be processed
    sentence_queue.join()
    audio_priority_queue.join()

    print(f"Fully processed this text: {full_result}")

    # Need to notify Openhab when we are done processing all audio.

if __name__ == "__main__":
    main()

