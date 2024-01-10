#%%
from pathlib import Path
from construct import max_
from openai import OpenAI, APIConnectionError
import queue, os
import threading
import subprocess
import pyaudio
from time import time, sleep
from decorators import run_in_thread
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type


# Initialize OpenAI client
client = OpenAI()
# Initialize a queue for text data
text_queue = queue.Queue()
play_queue = queue.Queue()
playing_until = 0


# talk system initialization flag
if 'initialized' not in globals():
    initialized = False

@run_in_thread
@retry(
    stop=stop_after_attempt(3),  # Stop after 3 attempts
    wait=wait_fixed(2),  # Wait for 2 seconds between retries
    retry=retry_if_exception_type(APIConnectionError)  # Retry on APIConnectionError
)
def text_to_speech(text, lang='hu'):
    # from gtts import gTTS
    # from io import BytesIO
    # tts = gTTS(text=text, lang=lang, slow=False)
    # audio_buffer = BytesIO()
    # tts.write_to_fp(audio_buffer)
    # audio_buffer.seek(0)

    tt = time()
    isready = threading.Event()
    mp3_path = Path(__file__).parent / "voice" / f"speech_{time():.2f}.mp3"
    add_play2queue(mp3_path, isready)
    # Create speech using OpenAI's TTS API
    response = client.audio.speech.create(
        model="tts-1",
        # model="tts-1-hd",
        voice="alloy",
        input=text
    )
    # Save the audio data to a file
    response.stream_to_file(mp3_path)
    isready.set()
    print("Resp:",time() - tt)

def play_audio_with_paplay(file_path, device_name=None):
    command = ['paplay']

    if device_name is not None:
        command += ['--device=' + device_name]

    command.append(file_path)

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while playing audio: {e}")
    remove_file(file_path)

def get_wav_duration(file_path):
    import wave
    with wave.open(str(file_path), 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration

@run_in_thread
def remove_file(filepath):
    # remove file on path
    try:
        os.remove(filepath)
        os.remove(filepath.with_suffix(".mp3"))
    except OSError as e:
        print(f"An error occurred while removing file: {e.filename} - {e.strerror}")

def process_play_queue():
    global playing_until
    while True:
        if not play_queue.empty():
            wav_path, isready = play_queue.get()
            isready.wait()
            play_audio_with_paplay(wav_path, 'Loopback_of_Discord')
        else:
            sleep(0.1)  # Sleep for a bit when the queue is empty

def wait_for_speech_sleep():
    sleep(1)
    ms_diff = playing_until - time() + 0.3  # for tolerance a little more.
    print(f"Sleeping for: {ms_diff:.1f}sec")
    if ms_diff > 0:
        sleep(ms_diff)

def process_text_queue():
    global playing_until
    max_text = ""
    while True:
        if not text_queue.empty():
            sleep(0.1) # a little wait, for things to get ready.
            while not text_queue.empty():
                max_text += text_queue.get()
            # Cut down the unfinished word
            # replace STEP 1 and STEP 2 and STEP 3: with empty string
            # replace CASE 1 and CASE 2 and CASE 3: with empty string
            # also handle : in the end of them
            max_text = max_text.replace('STEP 1:', '').replace('STEP 2:', '').replace('STEP 3:', '')
            max_text = max_text.replace('CASE 1:', '').replace('CASE 2:', '').replace('CASE 3:', '')
            max_text = max_text.replace('STEP 1', '').replace('STEP 2', '').replace('STEP 3', '')
            max_text = max_text.replace('CASE 1', '').replace('CASE 2', '').replace('CASE 3', '')
            j = 1
            while j<=len(max_text) and not max_text[-j] in ["?", "!", ",", ".", ":"]: # to only say whole sentences.
                j += 1
                if j == len(max_text):
                    break
            if j >= len(max_text):
                continue
            
            talk_text = max_text if j == 1 else max_text[:-j+1] 
            max_text = "" if j==1 else max_text[-j+1:]
            CHAR_PER_SEC = 15 # according to gpt its 12-14
            playing_until = max(time(), playing_until) + len(talk_text)/CHAR_PER_SEC # we need to know if it is already in the future.

            text_to_speech(talk_text)
            playing_until-time() > 0 and print(f'TTS: Sleep to not preprocess sentences too fast: {playing_until-time():.3f}', )
            sleep(max(playing_until-time()-0.5, 0.5))
        else:
            sleep(0.1)  # Sleep for a bit when the queue is empty

    
def init_talk_processor(force=False):
    global initialized
    if not force and initialized:
        return
    initialized = True
    print('initialized:', initialized)
    # Run the text processing in a separate thread
    t1 = threading.Thread(target=process_text_queue, daemon=True).start()
    threading.Thread(target=process_play_queue, daemon=True).start()

def add_text2queue(text):
    init_talk_processor()
    text_queue.put(text)

@run_in_thread
def add_play2queue(mp3_path, isready):
    wav_path = mp3_path.with_suffix('.wav')
    is_convready = threading.Event()
    play_queue.put((wav_path, is_convready))
    isready.wait()
    tt = time()
    convert_mp3_to_wav(mp3_path, wav_path)
    print("Conv:",time() - tt)
    init_talk_processor()
    is_convready.set()


def convert_mp3_to_wav(mp3_file_path, wav_file_path, speed=1.1):
    # from pydub import AudioSegment
    # audio = AudioSegment.from_mp3(mp3_file_path)
    # audio.export(wav_file_path, format="wav")
    ffmpeg_cmd = ['ffmpeg', '-y', '-i', f"{mp3_file_path}", '-filter:a', f'atempo={speed}', f"{wav_file_path}"]
    # Run the ffmpeg command
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def list_audio_devices():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        print(f"{i}: {dev['name']}", dev)
    p.terminate()

if __name__ == "__main__":
    add_text2queue("Figyelek! Jöhetnek a kérdések.")

    # Example: Adding text data to the queue
    # text_queue.put("Ez egy példaszöveg.")
    # text_queue.put("Még egy szöveg.")
    # sleep(10)

    # text_to_speech("Write me about the meaning of life. Once in a far far away galaxy, there was a sith, who controlled how things work in life, and he shouted! JUST DO IT! JUST DO IT! JUST DO IT! JUST DO IT!")
    # text_to_speech("így beszélne az AI ha tudna a mobilon keresztül beszélni.")
    # text_to_speech("Mondjuk mondhatná ezt, vagy amire éppen szükségünk van.")
    # add_text2queue("Igen eléggé")
    # sleep(0.1)
    # add_text2queue("érdekes az akcentusa,")
    # sleep(0.1)
    # add_text2queue("viszont legalább tud beszélni.")
    # list_audio_devices()
    # mp3_path = Path(__file__).parent / "voice" / f"speech_1703116391.68.mp3"
    # wav_path = Path(__file__).parent / "voice" / f"speech_1703116391.68.wav"
    # tt = time()
    # play_audio_with_paplay(wav_path, 'Loopback_of_Discord')
    # convert_mp3_to_wav(mp3_path, wav_path)
    # print(time() - tt)