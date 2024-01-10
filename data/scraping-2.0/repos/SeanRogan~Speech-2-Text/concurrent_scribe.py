import glob
import os
import time
import openai
import constants
import logging
import concurrent.futures


# This script will work, but I have yet to determine if there is any noticeable performance benefit from using the concurrent version.
logging.basicConfig(level=logging.DEBUG, filename='log.txt')
last_processed_time = 0  # last processed file timestamp
transcribed_files = []  # list to hold files already transcribed to avoid duplicate api calls


def transcribe(path):
    logging.debug('transcribe proc started' + str(time.perf_counter()))
    with open(path, 'rb') as file:
        logging.debug('file opened' + str(time.perf_counter()))
        logging.debug('calling api with file: ' + str(latest))
        api_call_result = openai.Audio.transcribe("whisper-1", file, api_key=constants.OPENAI_API_KEY)  # OpenAI API call, args include whisper model, sending the file object,and the api key
        logging.debug("api call success")
        logging.debug(str(api_call_result))
        logging.debug('api call took ' + str(time.perf_counter()) + 'from file open to result')

        print(str(api_call_result.text))  # print the results of the api call

    with open('transcription.txt', 'a') as f:
        f.write(api_call_result.text)
        # record file timestamp as new last_processed_file


while True:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        files = glob.glob("./recordings/*")  # files are in the recordings dir
        unprocessed_files = [f for f in files if os.path.getctime(f) > last_processed_time]  # unprocessed_files are any with a timestamp more recent than the last_processed_file's timestamp
        if not unprocessed_files:
            time.sleep(1)  # if there aren't any files to process, sleep 1second and restart loop
            continue

        latest = max(unprocessed_files, key=os.path.getctime)  # sort unprocessed files by timestamp to find the latest recording

        logging.debug(str(latest))
        latest_filename = latest.split('/')[2]  # split file id from dir id
        logging.debug("latest filename is : " + str(latest_filename))

        if latest not in transcribed_files and os.path.exists(latest):  # if file path exists and is not present in the 'transcribed' list of files already processed..
            future = executor.submit(transcribe, latest)  # transcribe thread spawned
            result = future.result()  # thread returns file path to be recorded once processing is finished
            transcribed_files.append(result)
            last_processed_time = os.path.getctime(latest)
