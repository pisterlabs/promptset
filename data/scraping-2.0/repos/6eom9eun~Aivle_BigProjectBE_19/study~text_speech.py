from gtts import gTTS
from openai import OpenAI
from pathlib import Path
import os
import json

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR/'secrets.json') as f:
    secrets = json.loads(f.read())

os.environ['OPENAI_API_KEY']= secrets['OPENAI_API_KEY']

#----------------정민권 수정 사항 (아래)--------
# !pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# !pip install git+https://github.com/sanchit-gandhi/whisper-jax.git
# !pip install cached-property
# !pip install ipython
# !pip install torch

from IPython.display import Audio, display
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
from django.http import HttpResponse

# # 텍스트를 음성으로 변환(TTS)
# def Text_To_Speech(sentence):
#     # JSON 파일에서 OCR 결과 로드(결과 파일 경로 입력)
#     json_file_path = ".../ocr_results.json"

#     #지정된 경로로 json 파일 로드
#     with open(json_file_path, "r", encoding="utf-8") as json_file:
#         ocr_results = json.load(json_file)
    
#     # 단락을 음성으로 변환 후 저장
#     for i, paragraph in enumerate(ocr_results, start=1):
#     # 언어 설정(기본 값은 'en', 한국어는 'ko')
#         tts = gTTS(text=paragraph, lang="ko",slow=False)
#         mp3_file_path = f".../speech_{i}.wav"     #(음성 파일 결과 경로 입력)
#     # 음성 파일로 저장
#         tts.save(mp3_file_path)
    

# 음성을 텍스트로 변환(STT)


# def flush():
#   gc.collect()           # 파이썬 가비지 컬렉션을 수행하여 메모리를 정리합니다.
#   torch.cuda.empty_cache() # PyTorch의 CUDA 캐시를 비웁니다

# import librosa

# def Speech_To_Text(file_path):
#     client = OpenAI()

#     audio_data, _ = librosa.load(file_path, sr=16000)

#     pipeline = FlaxWhisperPipline("openai/whisper-small", dtype=jnp.float16)
#     transcript = pipeline(audio_data, return_timestamps=True)

#     return transcript

# 텍스트를 음성으로 변환
# def Text_TO_Speech(sentence):
#     # JSON 파일에서 OCR 결과 로드(결과 파일 경로 입력)
#     json_file_path = ".../ocr_results.json"
    
#     # 언어 설정(기본 값은 'en', 한국어는 'ko')
#     language='ko'
    
#     # gTTS 객체 생성
#     tts = gTTS(text=sentence, lang=language, slow=False)
    
#     # 음성 파일로 저장
#     tts.save("output.mp3")
    

# 음성을 텍스트로 변환
def Speech_To_Text(file_path):
    client=OpenAI()
    
    audio_file=open(file_path,'rb')
    
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file).text
    print(transcript)
    
    return transcript


# 코사인 유사도를 통한 채점 시스템
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response_is_correct(text1,text2):
    # CountVectorizer를 사용하여 문장을 벡터로 변환
    vectorizer = CountVectorizer().fit_transform([text1, text2])

    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(vectorizer)
    print(cosine_sim[0][1])
    
    if cosine_sim[0][1]>=0.3:
        return True
    else:
        return False