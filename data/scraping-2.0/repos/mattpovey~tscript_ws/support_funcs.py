
import re
import os
import sys
import openai
import tiktoken
import logging
import logging.handlers
import subprocess
from flask import jsonify
from pydub import AudioSegment
import imageio_ffmpeg as ffmpeg

# List of system roles to be added to messages
sys_roles = []
sys_roles.append({"role": "mtg_notes", "content": "You are an assistant with the job of summarizing what was discussed in meetings."})
sys_roles.append({"role": "iview_summary", "content": "You are an interviewer with the job of transcribing an interview so that questions and answers are identified clearly."})
sys_roles.append({"role": "pod_summary", "content": "You summarize the podcast transcript to highlight what's most interesting"})
sys_roles.append({"role": "expand", "content": "You are a copywriter who needs to expand a short piece of text to make it longer. Build on the points in the original text and make sure the new text is still on topic."})
sys_roles.append({"role": "test", "content": "This is a test. Resopnd with the first sentence of the text provided and the word TEST."})
role_msgs = []

openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a logger
logger = logging.getLogger('tscript_logger')
logger.setLevel(logging.DEBUG)

# Create a SysLogHandler
# set the log location based on OS
if sys.platform.startswith('win'):
    syslog_handler = logging.handlers.SysLogHandler(address=('localhost', 514), socktype=socket.SOCK_STREAM)    
elif sys.platform.startswith('linux'):
    syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
else:
    logger.addHandler(logging.handlers.SysLogHandler())

# Count tokens using the OpenAI tiktoken library
def tok_count(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    n_toks = len(encoding.encode(text))
    return n_toks

# Chunk text to fit inside token window
# If I ever get diarization working, this will need to be updated to handle speaker changes
def split_text(text, max_tokens):
    words = re.split(r'\s+', text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    i = 0
    while i < len(words):
        word = words[i]
        word_token_count = tok_count(word)
        current_token_count += word_token_count

        if current_token_count > max_tokens:
            # Find the nearest full stop after the token limit
            while "." not in word and i < len(words) - 1:
                i += 1
                word = words[i]
                current_token_count += tok_count(word)

            # Split at the full stop
            before_full_stop, after_full_stop = word.split(".", 1)
            current_chunk.append(before_full_stop + ".")
            chunks.append(" ".join(current_chunk))

            # Start a new chunk with the remaining part of the split word
            current_chunk = [after_full_stop.lstrip()]
            current_token_count = tok_count(after_full_stop)
        else:
            current_chunk.append(word)
        i += 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def gpt_proc(text, sys_role, gpt_model="gpt-3.5-turbo", remember="true"):
    print("System role is: " + sys_role)
    role_msgs = [] # List to store the role messages

    # The max_tokens changes based on the model and how much we need to process
    # If we have more than the max_tokens for the model, we need to process in chunks
    # Dictionary of models and their max tokens
    model_max_tokens = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-32k": 16384
    }

    # Get the number of chunks based on model size. We only need summary tracker space
    # if we have more content than the model can handle in one go.
    max_chunks = split_text(text, model_max_tokens[gpt_model])
    # Set the max tokens based on whether we are using remember or not
    # Leave ~3000 tokens for the summary
    # Update values in the dictionary accordingly
    if remember == "true" and len(max_chunks) > 1:
        max_tokens = model_max_tokens[gpt_model] - 3000
        print("Max tokens: " + str(max_tokens))
    else:
        max_tokens = model_max_tokens[gpt_model]
        print( "Max tokens: " + str(max_tokens))

    # Configure the role messages. 
    # First get the system role content based on argument
    for i in sys_roles:
        if i["role"] == sys_role:
            sys_role_content = i["content"]
            break
        else:
            pass

    if sys_role_content == "":
        print("No system role content found.")
        logger.info("gpt_proc(): No system role content found. Processing stopped.")
        return "No system role content found. Processing stopped."

    # split_text() will split the text into chunks that fit within the token limit
    chunks = split_text(text, max_tokens)
    print("There are", len(chunks), "chunks.")

    # Keeps track of the summary
    summary = ""

    # Makes sure that previous summaries are considered.
    sum_tracker = []
    n=1
    for i in chunks:
        # Configure the role message
        # If the model is gpt4 family, the system role is added to the first message
        # Otherwise, it is appended as the final message for each chunk.
        if gpt_model == "gpt-4" or gpt_model == "gpt-4-32k":
            role_msgs.append({"role": "system", "content": sys_role_content})
            # Add the current chunk to the role_msgs list
            role_msgs.append({"role": "user", "content": i})
        elif gpt_model != "gpt-4" and gpt_model != "gpt-4-32k":
            # Add the current chunk to the role_msgs list
            role_msgs.append({"role": "user", "content": i})
            # Add the system role content to the role_msgs list
            role_msgs.append({"role": "system", "content": sys_role_content})

        # If we're doing more than one run, add the previous summary to the role_msgs list
        # There will only ever be content in sum_tracker if remember is true
        print("Remember is: " + remember)
        if remember == "true" and len(sum_tracker) > 0:
            # Add the previous summaries to the role_msgs list
            print("There are", len(sum_tracker), "items in the summary tracker.")
            print("Adding previous summaries to role_msgs list.")
            # Only append the content, not the list
            for j in sum_tracker:
                if gpt_model == "gpt-4" or gpt_model == "gpt-4-32k":
                    role_msgs.append(j)
                elif gpt_model != "gpt-4" and gpt_model != "gpt-4-32k":
                    for k in j:
                        # Insert at the start of the list in reverse order
                        n = len(k)
                        while n > 0:
                            role_msgs.insert(0, k[n-1])
                            n -= 1

        print("Run: " + str(n) + " of " + str(len(chunks)) + " New tokens: " + str(tok_count(i)))
        logger.info("Run: " + str(n) + " of " + str(len(chunks)) + " New tokens: " + str(tok_count(i)))
        try:
            response = openai.ChatCompletion.create(
                model=gpt_model,
                messages=role_msgs,
            )
        except openai.error.APIError as e:
            print(f"An API error occurred: {e}")
            logger.info (f"An API error occurred: {e}")
            raise e
        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            logger.info (f"Failed to connect to OpenAI API: {e}")
            raise e
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            logger.info (f"OpenAI Rate limit exceeded: {e}")
            raise e
        except openai.error.AuthenticationError as e:
            print(f"Authentication error: {e} \n")
            print("Check your OpenAI API key is provided and correct. \n")
            logger.info (f"Authentication error: {e} \n")
            # jsonification is performed by the caller.
            raise e
        
        print("Response received.")
        r_text = response['choices'][0]['message']['content']
        print("Partial summary for chunk " + str(n) + ": " + r_text)
        summary += "\n" + r_text + "\n"

        # Help the LLM know what was discussed earlier in the meeting.
        # Only add to the summary tracker if remember is true
        # TODO - Flesh this out into a proper approach to dealing with larger inputs

        if remember == "true":
            tracker_msg = "This is the summary of what happened earlier: " + r_text
            sum_tracker.append({"role": "assistant", "content": tracker_msg})
        n+=1
        # Delete all items from role_msgs except the first one
        # del role_msgs[1:]
    return summary

def rename_m4a(file_path, new_ext='mp4'):
    # ffmpeg seems to have problems with m4a files which are resolved if they 
    # are renamed to mp4 first
    print("Renaming M4A to MP4")
    new_file_path = file_path.rsplit('.', 1)[0] + "." + new_ext
    os.rename(file_path, new_file_path)
    return new_file_path

# Split audio into chunks of 25MB
# The OpenAI API has a limit of 25MB per request
def split_audio(file_path, max_size_bytes=25 * 1024 * 1024, file_ext=None):
    print("In split_audio(), file path: " + file_path)
    print("In split_audio(), file extension: " + file_ext)

    file_size = os.path.getsize(file_path)

    if file_size <= max_size_bytes:
        print("File size is less than max size")
        return [AudioSegment.from_file(file_path, format=file_ext)]

    audio_formats = {'mp3': 'from_mp3', 'mp4': 'from_file', 'm4a': 'from_file', 'wav': 'from_wav'}
    audio_func = audio_formats.get(file_ext, 'from_wav')
    print("Audio function: " + audio_func)
    audio = getattr(AudioSegment, audio_func)(file_path)

    chunk_length = int((len(audio) * max_size_bytes) / file_size)
    print("Chunk length: " + str(chunk_length))
    
    #audio_chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
    audio_chunks = []
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i+chunk_length]
        audio_chunks.append(chunk)
    audio_segments = []

    for chunk in audio_chunks:
        audio_segments.extend(chunk.split_to_mono())

    return audio_segments

def convert_to_wav(input_file_path: str, output_wav_file_path: str):
    input_file_ext = os.path.splitext(input_file_path)[-1].lower()
    print("Input file extension: " + input_file_ext)
    if input_file_ext != ".wav":
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        ffmpeg_cmd = [
            ffmpeg_exe, "-i", input_file_path,
            "-y", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            output_wav_file_path
        ]
        try:
            print("Converting to WAV with ffmpeg:")
            ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            print(ffmpeg_result.stdout)
        except subprocess.CalledProcessError as e:
            raise Exception("FFmpeg conversion failed", e.stderr)

def process_transcription(input_wav_file_path: str, options, output_file_path: str):
    main_cmd = ["./whisper.cpp/main"]

    if options.get('model'):
        model_path = f"./whisper.cpp/models/ggml-{options['model']}.bin"
        main_cmd.extend(["-m", model_path])

    if options.get('translate'):
        main_cmd.append("-tr")

    if options.get('language'):
        main_cmd.extend(["-l", options['language']])

    if options.get('outfmt') == "txt":
        main_cmd.append("-otxt")
    elif options.get('outfmt') == "srt":
        main_cmd.append("-osrt")
    else:
        raise Exception("Invalid output format")

    main_cmd.extend(["-of", os.path.splitext(output_file_path)[0]])
    main_cmd.append(input_wav_file_path)
    try:
        main_result = subprocess.run(main_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception("Transcription failed", e.stderr)

def main():
    # with open("tscript.txt", "r") as f:
    #     text = f.read()

    # role="mtg_notes"

    # summary = gpt_proc(text, sys_role=role, max_tokens=1000, remember="false")
    # print("-----------\nSummary:\n-----------")
    # print(summary)
    print("This is a library. Nothing to run here.")

if __name__ == "__main__":
    main()