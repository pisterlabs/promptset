import os
import time
import re
from flask import Flask, request, redirect
import whisper
import threading
import openai
from requests.exceptions import Timeout

openai.api_key = open("api_key").read().strip()

PORT = 9999
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'m4a','mp3','wav'}
USE_WHISPER_API = True
WHISPER_MODEL_NAME = 'small' # tiny, base, small, medium
WHISPER_API_MODEL_NAME = 'whisper-1'
WHISPER_DEVICE = 'cpu' # cpu, cuda
CHATGPT_MODEL_NAME = 'gpt-3.5-turbo-0613'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__, static_url_path='/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

lock = threading.Lock()

if not USE_WHISPER_API:
   print('loading whisper model', WHISPER_MODEL_NAME, WHISPER_DEVICE)
   whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
prev_transcript = "" # whisper API使用時にpromptとして前回のtranscriptionを入れる
pended_translation = "" # APIのtimeout等で翻訳をskipしたテキストを保持しておく


@app.route('/')
def index():
   return redirect('/index.html')


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    file = request.files['file']
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext and ext in ALLOWED_EXTENSIONS:
        filename = str(int(time.time())) + '.' + ext
        saved_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(saved_filename)
        with lock:
            try:
                result = trascribe_by_whisper(saved_filename)
                translation = translate_by_chatgpt(result.text)
                if translation == "":
                    translation = "\n"
            except Exception as e:
                print(e)
                result = {'error': 'something wrong'}
                return result, 400
            else:
                result.translation = translation
                return result, 200

    result = {'error':'wrong format of file'}
    print(result)
    return result, 400


def trascribe_by_whisper(saved_filename):
    time_sta = time.perf_counter()
    print('start transcribe ' + str(time_sta))
    if USE_WHISPER_API:
        audio_file = open(saved_filename, "rb")
        # 精度を上げるため前回のtranscriptionをpromptとして渡す
        global prev_transcript
        result = openai.Audio.transcribe(WHISPER_API_MODEL_NAME, audio_file, prompt=prev_transcript, language='en')
        prev_transcript = result.text
        print(f"whisper API responce: '{result.text}'")
    else:
        result = whisper_model.transcribe(saved_filename, beam_size=3, fp16=False, language='en')
        print(result)

    print('whisper response time: ' + str(time.perf_counter() - time_sta))
    return result


def translate_by_chatgpt(transcription):
    global pended_translation

    to_translate = extract_texts_to_translate(transcription)
    if to_translate == "":
        return "" # 翻訳できるものがないので空文字を返す
    time_sta = time.perf_counter()
    print(f"started to translate '{to_translate}'")

    prompt = f"""
        以下は自然言語処理の国際学会での英語のスピーチの一部です。日本語に翻訳したものだけを出力してください
        {to_translate}
    """
    try:
        response = openai.ChatCompletion.create(
            model=CHATGPT_MODEL_NAME,
            messages=[
                {"role": "system", "content": "あなたは翻訳機です"},
                {"role": "user", "content": prompt},
            ],
            top_p=0.1,
            request_timeout=10,
        )
    except Timeout as e:
        print(e)
        pended_translation = f"{to_translate} {pended_translation}"
        return ""
    except Exception as e:
        raise e
    else:
        print("chatgpt response: '", response["choices"][0]["message"]["content"], "'")
        print('chatgpt response time: '+ str(time.perf_counter() - time_sta))
        return response["choices"][0]["message"]["content"]


def extract_texts_to_translate(transcription):
    """
    transcriptionのうち文が完結している部分だけ抽出し残りはpending_translationに保持
    """
    global pended_translation
    # 文等の空白を削除
    transcription = re.sub(r'^\s+', '', transcription)
    # 文の途中で終わっている場合末尾に...がつくことがあるので削除
    if transcription.endswith('...'):
        transcription = transcription[:-3]

    # transcriptionの先頭が大文字で始まっていない場合は文の終わりでないとみなしpending_translationの末尾の句読点を削除
    eos_markers = ['.', '?', '!']
    starts_with_upper = transcription[0].isupper()
    pended_ends_with_eos = any(pended_translation.endswith(marker) for marker in eos_markers)
    if pended_ends_with_eos and not starts_with_upper:
        pended_translation = pended_translation[:-1]
    
    whole_texts = f"{pended_translation} {transcription}"

    if re.search(r'[\.\?\!]\s+(?=\S)', whole_texts):
        pended_translation = re.split(r'[\.\?\!]\s+(?=\S)', whole_texts)[-1]
        to_translate = whole_texts[:-len(pended_translation)]
    else:
        pended_translation = whole_texts
        to_translate = ""

    return to_translate


if __name__ == '__main__':
    app.run(host='localhost', port=PORT)
