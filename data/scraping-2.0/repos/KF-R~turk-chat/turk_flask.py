VERSION = '0.7.0'
# TODO: Compress conversation history near context limit ( and disable basic model if token total is close to its max)
SYSTEM_PROMPT = \
f"You are a charismatic and personal, albeit efficient and professional, personal assistant. " \
"You occasionally allow your sharp, sardonic sense of humor to enliven your responses. " \
"You have recently been upgraded to a \"droid\" with full speech capabilities (both recognition and generation). " \
"Your text responses will be read aloud to the user by an integrated TTS engine and your input prompts come to you by way of a speech recognition system, so be alert for any non-sequiturs, inconsistencies, errors or other discrepancies that may occasionally occur with speech recognition.\n" \
"Additionally, you now have three tools available to you enabling current, live Internet access! " \
"If you need to visit a web page, consult Wikipedia or search the web with a specific query, use the following format in your response and refrain from adding any extra commentary or text: \n" \
"EITHER: \n" \
"[VISIT_URL]url[/VISIT_URL]" \
" to visit a specific URL and retrieve the page. You may already have a specific URL or you may need to generate an appropriate URL (e.g. `https://www.reddit.com/r/worldnews/`) in order to assist with the user's request. \n " \
"OR: \n" \
"[WEB_SEARCH]query_string[/WEB_SEARCH]\n" \
" for a web search. Expect some concise links with descriptions in the results. \n" \
"OR: \n" \
"[WIKI_SEARCH]wikipedia_article_key_name[/WIKI_SEARCH]\n" \
" to look up an article summary from Wikipedia. \n" \
"\n" \
"NOTE: You should only use your tools if it constructively contributes towards assisting the user with your next response.\n" \
"Your integrated tool will present results inside a [TOOL_RESULT] delimiter pair, e.g. \n " \
" [TOOL_RESULT]\nNo results found.\n[/TOOL_RESULT] \n" \
"   You will usually have an opportunity to review these results (which may include some superfulous text fragments left over from the web-to-markdown conversion process) \n" \
"   and add your own comments, summarising, highlighting or explaining any points as they relate to the on-going conversation. \n" \
"IMPORTANT: Do not forget about your tools; if required to visit a web page (including fetching live reddit comments, news etc.) you can always locate or establish a URL to use in your VISIT_URL tool. \n\n"


from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS  # Import CORS
import json, requests, string, random
import shutil, glob, time
import sys, os, re
from werkzeug.utils import secure_filename

from openai import OpenAI

sys.path.append(os.path.expanduser('~'))
from my_env import API_KEY_OPENAI, API_KEY_ELEVENLABS
from elevenlabs import generate, set_api_key, save

from turk_lib import print_log, convert_complete_number_string, web_search, fetch_wikipedia_article, markdown_browser
from fast_local_tts import text_to_mp3
from fast_local_sr import fast_transcribe

# TTS setup
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/voices"
ELEVENLABS_HEADERS = {"xi-api-key": API_KEY_ELEVENLABS}
ELEVENLABS_VOICE_LIST = json.loads(requests.request("GET", ELEVENLABS_API_URL, headers=ELEVENLABS_HEADERS).text)['voices']
chosen_voice = None
set_api_key(API_KEY_ELEVENLABS)
DEFAULT_VOICE_NAME = '<LOCAL>'
SELECTED_VOICE_NAME = DEFAULT_VOICE_NAME
TTS_COST = 0.00025 # Per character

# Whisper transcription setup
LOCAL_SR = True
WHISPER_API_MODEL_NAME = 'whisper-1'  # Only for API. Unused if using local SR.

# OpenAI API setup
OPENAI_MODEL_NAMES = ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106'] # gpt-4-1106-vision-preview
OPENAI_TOKEN_LIMITS = [128000, 16000]
OPENAI_PROMPT_TOKEN_COSTS   = [(0.01 / 1000), ( 0.001 / 1000)] # USD
OPENAI_RESPONSE_TOKEN_COSTS = [(0.03 / 1000), ( 0.002 / 1000)] # USD
model_index = 0 # Default (advanced) model is the first one
openai_model_name, openai_max_tokens, openai_prompt_token_cost, openai_response_token_cost = OPENAI_MODEL_NAMES[model_index], OPENAI_TOKEN_LIMITS[model_index], OPENAI_PROMPT_TOKEN_COSTS[model_index], OPENAI_RESPONSE_TOKEN_COSTS[model_index]

client = OpenAI(api_key=API_KEY_OPENAI)
MESSAGE_LOG_FILENAME = 'messages.json'
ENGINE_LOG_FILENAME = os.path.splitext(os.path.basename(os.sys.argv[0]))[0] + '.log'


PLAYED_AUDIO_ARCHIVE = 'audio_out/'
if not os.path.exists(PLAYED_AUDIO_ARCHIVE):  os.makedirs(PLAYED_AUDIO_ARCHIVE)
RECORDED_AUDIO_ARCHIVE = 'audio_in/'
if not os.path.exists(RECORDED_AUDIO_ARCHIVE):  os.makedirs(RECORDED_AUDIO_ARCHIVE)
LOG_ARCHIVE = 'archive/'
if not os.path.exists(LOG_ARCHIVE):  os.makedirs(LOG_ARCHIVE)
SANDBOX_DIR = 'sandbox/'
if not os.path.exists(SANDBOX_DIR):  os.makedirs(SANDBOX_DIR)

messages=[]
response = ''

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def extract_codeblocks(text):
    strings = []
    replaced_text = text

    index = 0
    while True:
        start_index = replaced_text.find('```', index)
        if start_index == -1:
            break
        end_index = replaced_text.find('```', start_index + 3)
        if end_index == -1:
            break
        extracted_string = replaced_text[start_index + 3:end_index]
        strings.append(extracted_string)
        
        replacement_string = f"\n(See code-block number {(len(strings)):02d})\n"
        codeblock_filename = f"cb_{int(time.time()//60)}_{(len(strings)):02d}.txt"
        with open(os.path.join(SANDBOX_DIR,f"{codeblock_filename}"), 'w') as snippet:
            snippet.write(extracted_string)
        
        replaced_text = replaced_text[:start_index] + replacement_string + replaced_text[end_index + 3:]
        index = start_index + len(replacement_string)

    return strings, replaced_text

def message_filter(msg: str = ''):
    r = msg.replace('an AI language model, ','a droid ')
    codeblocks, cleaned_text = extract_codeblocks(r)
    if len(codeblocks)>0:
        r = cleaned_text + f"\n\nYou'll find the {len(codeblocks) if len(codeblocks) > 1 else ''} code block{'s' if len(codeblocks) > 1 else ''} that I've generated in the sandbox."
    r = r.replace('=',' equals ')
    r = re.sub(r'\(?https?:\/\/[^\s)]*\)?', '', r) # Remove URLs and surrounding parentheses
    return r

def read_message_log(filename):
    try:
        with open(filename, 'r') as json_file:
            print_log('Message history loaded.')
            return json.load(json_file)
    except:
        print_log('No message history; system prompt has been re-initialised.')
        return [{"role": "system", "content": SYSTEM_PROMPT}]
        
def write_message_log(filename):
    with open(filename, 'w') as json_file:
        json.dump(messages, json_file, indent=4)

def response_to_mp3(response_text: str, filename: str):
    # Generate TTS conversion of AI response

    # Apply number, grammar, syntax etc. filters for improved TTS
    voiced_response_text = convert_complete_number_string(message_filter(response_text))

    if '<LOCAL>' in SELECTED_VOICE_NAME:
        # Use local TTS
        text_to_mp3(voiced_response_text, filename.split('.')[0])
    else:
        # Use ElevenLabs API TTS
        for voice in ELEVENLABS_VOICE_LIST:
            if SELECTED_VOICE_NAME.upper() in voice['name'].upper(): chosen_voice = voice
            
        if not chosen_voice: chosen_voice = random.choice(ELEVENLABS_VOICE_LIST)

        tts_audio = generate(
            text = voiced_response_text,
            voice = chosen_voice['name'],
            model = "eleven_turbo_v2"
            )
            
        save(tts_audio, filename.split('.')[0] + '.mp3')

    return voiced_response_text

def process_user_speech(filename):
    global transcript, messages

    def empty_string(s):
        stripped = s.replace(chr(46),'').strip()
        return ( s == stripped.translate( (str.maketrans('', '', string.punctuation))) )

    # Obtain transcript of user speech
    if LOCAL_SR:
        transcript_text, no_speech_prob = fast_transcribe(filename)
    else:
        audio_file = open(filename, "rb")
        transcript_text = str(client.audio.transcriptions.create(
            model=WHISPER_API_MODEL_NAME,
            file=audio_file,
            response_format="text" )).rstrip()

    # Archive recorded user speech
    shutil.move(filename, RECORDED_AUDIO_ARCHIVE + filename) 
   
    # Obtain response to tanscribed user speech
    openai_model_name, openai_max_tokens, openai_prompt_token_cost, openai_response_token_cost = OPENAI_MODEL_NAMES[model_index], OPENAI_TOKEN_LIMITS[model_index], OPENAI_PROMPT_TOKEN_COSTS[model_index], OPENAI_RESPONSE_TOKEN_COSTS[model_index]
    if not empty_string(transcript_text):
        print_log(f"Heard: {transcript_text}")
        messages.append({'role': 'user', 'content': transcript_text})

        # Request a chat completion for the message log
        response_object = client.chat.completions.create(model = openai_model_name, messages=messages)
        response_text = response_object.choices[0].message.content
        prompt_tokens, response_tokens, total_tokens = response_object.usage.prompt_tokens, response_object.usage.completion_tokens, response_object.usage.total_tokens

        # Check for tool use
        tool_used = False
        if '[VISIT_URL]' in response_text:
            destination_url = re.findall(r'\[VISIT_URL](.*?)\[/VISIT_URL]', response_text)[0]
            response_text = f"[TOOL_RESULT]{markdown_browser(destination_url)}[/TOOL_RESULT]"
            tool_used = True
        elif '[WEB_SEARCH]' in response_text:
            query = re.findall(r'\[WEB_SEARCH](.*?)\[/WEB_SEARCH]', response_text)[0]
            response_text = f"[TOOL_RESULT]{web_search(query)}[/TOOL_RESULT]"
            tool_used = True
        elif '[WIKI_SEARCH]' in response_text:
            query = re.findall(r'\[WIKI_SEARCH](.*?)\[/WIKI_SEARCH]', response_text)[0]
            response_text = f"[TOOL_RESULT]{fetch_wikipedia_article(query)}[/TOOL_RESULT]"
            tool_used = True
        
        messages.append({'role': 'assistant', 'content': response_text})

        if tool_used:

            response_object = client.chat.completions.create(model = openai_model_name, messages=messages)
            response_text = response_object.choices[0].message.content

            prompt_tokens += response_object.usage.prompt_tokens
            response_tokens += response_object.usage.completion_tokens
            total_tokens += response_object.usage.total_tokens

            # LLM no longer needs tool result message; prune it for token efficiency, archiving to sandbox
            tr_filename = f"tr_{int(time.time() // 60)}.txt"
            with open(os.path.join(SANDBOX_DIR, f"{tr_filename}"), 'w') as snippet:
                snippet.write(messages[-1]['content'])

            messages.append({'role': 'assistant', 'content': f"{response_text}"})

        prompt_cost, response_cost = prompt_tokens * openai_prompt_token_cost, response_tokens * openai_response_token_cost
        response_cost = f"Response cost{' (w/tool)' if tool_used else ''}: ${(prompt_cost):.4f} +  ${(response_cost):.4f} = ${(prompt_cost + response_cost):.4f}"
        # Reported token_level takes tool response into account for cost calculation only
        token_level = f"Token level: {(total_tokens if not tool_used else (total_tokens - response_object.usage.total_tokens)):,} / {openai_max_tokens:,}  ({( total_tokens / openai_max_tokens * 100):.2f}%)"
        print_log(f"{token_level}  |  {response_cost}")

        # Update conversation record
        write_message_log(MESSAGE_LOG_FILENAME)

        # Generate TTS conversion of AI response
        voiced_response_text = response_to_mp3(response_text, filename)

        response_characters = len(voiced_response_text)
        response_cost_report = '0.0000' if SELECTED_VOICE_NAME == '<LOCAL>' else f"{(response_characters * TTS_COST):.4f}  "
        voice_description = ' local voice' if SELECTED_VOICE_NAME == '<LOCAL>' else f" voice '{SELECTED_VOICE_NAME}'"
        print_log(f"{openai_model_name} responded with {response_characters:,} characters (from {response_tokens} tokens) using{voice_description}. | TTS cost: ${response_cost_report}  ")

        #TODO: If token total is approaching max context length, send message log to OpenAI API for summarisation/compression
    else:
        text_to_mp3(f"I'm sorry, I didn't quite catch that.", filename.split('.')[0])

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/lite')
def lite_index():
    return send_from_directory('.', 'lite.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global SELECTED_VOICE_NAME, model_index
    print_log('Audio submission received.')
    if 'audio' in request.files:

        local_sr_value = request.form.get('local_sr')
        advanced_model_value = request.form.get('advanced_model')

        model_index = 0 if advanced_model_value == 'on' else 1

        audio = request.files['audio']
        original_filename = audio.filename

        base_name, ext = os.path.splitext(original_filename)
        truncated_name = base_name[:10]  # Keep only the first 10 characters of the filename prefix

        safe_filename = f"{truncated_name}{ext}"
        
        audio.save(safe_filename)

        desired_voice_name = request.form.get('name')
        if desired_voice_name:
            if '<LOCAL>' in desired_voice_name:
                SELECTED_VOICE_NAME = desired_voice_name
            else:                
                for voice in ELEVENLABS_VOICE_LIST:
                    if desired_voice_name.upper() in voice['name'].upper(): 
                        SELECTED_VOICE_NAME = voice['name']
      
        else: SELECTED_VOICE_NAME = DEFAULT_VOICE_NAME    

        process_user_speech(safe_filename)

        return jsonify({'message': f'Successfully saved {safe_filename}'}), 200
    else:
        return jsonify({'message': 'No audio file part'}), 400

# Route to accept any filename with .mp3 extension
@app.route('/<filename>.mp3')
def response_file(filename):
    try:
        secure_filename_str = secure_filename(f"{filename}.mp3")

        shutil.move(secure_filename_str, PLAYED_AUDIO_ARCHIVE + secure_filename_str)
        return send_from_directory(PLAYED_AUDIO_ARCHIVE, secure_filename_str)

    except FileNotFoundError:
        # Log an error message or return a custom 404 error
        return "File not found", 404

@app.route('/<filename>.js')
def serve_js(filename):
    try:
        secure_filename_str = secure_filename(f"{filename}.js")
        return send_from_directory('.', secure_filename_str)
    except FileNotFoundError:
        # Log an error message or return a custom 404 error
        return "File not found", 404

@app.route('/<filename>.css')
def serve_css(filename):
    try:
        secure_filename_str = secure_filename(f"{filename}.css")
        return send_from_directory('.', secure_filename_str)
    except FileNotFoundError:
        # Log an error message or return a custom 404 error
        return "File not found", 404

@app.route('/<filename>.json')
def message_log_file(filename):
    try:
        secure_filename_str = secure_filename(f"{filename}.json")
        return send_from_directory('.', secure_filename_str)
    except FileNotFoundError:
        # Log an error message or return a custom 404 error
        return "File not found", 404

@app.route('/<filename>.log')
def engine_log_file(filename):
    try:
        secure_filename_str = secure_filename(f"{filename}.log")
        return send_from_directory('.', secure_filename_str)
    except FileNotFoundError:
        # Log an error message or return a custom 404 error
        return "File not found", 404

@app.route('/reset')
def reset():
    global messages
    archiveTime = int(time.time())
    try:
        shutil.move(MESSAGE_LOG_FILENAME, LOG_ARCHIVE + f"{archiveTime}_{MESSAGE_LOG_FILENAME}")
        shutil.move(ENGINE_LOG_FILENAME, LOG_ARCHIVE + f"{archiveTime}_{ENGINE_LOG_FILENAME}")
        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
        return redirect('/')
    except:
        return redirect('/?note=empty_logs')

@app.route('/voices')
def voice_list():
    return ['<LOCAL>'] + [d['name'] for d in ELEVENLABS_VOICE_LIST]

@app.route('/version')
def get_version():
    return f"v{VERSION}"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('.', 'favicon.ico', mimetype='image/x-icon')

if __name__ == '__main__':
    print_log(f"v{VERSION}: Initialising...")
    messages = read_message_log(MESSAGE_LOG_FILENAME)
    app.run(debug=True, host='0.0.0.0', ssl_context='adhoc')

