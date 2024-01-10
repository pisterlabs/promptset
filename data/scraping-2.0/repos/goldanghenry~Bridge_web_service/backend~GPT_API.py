from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
# from pydub import AudioSegment
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)
app.config['TEMPLATES_AUTO_RELOAD'] = True  # 템플릿 자동 리로드 활성화

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    # .env에 있는 OPENAI_API_KEY 가져옴 

# OpenAI GPT-3 API 엔드포인트와 API 키를 설정합니다.
# OPENAI_API_ENDPOINT = "https://api.openai.com/v1/completions"
client = OpenAI()
client.api_key = OPENAI_API_KEY

# 음성 파일 저장 경로
upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

@app.route('/')
def index():
    result = 'stt result :'
    return render_template('index.html', result=result)

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 동적으로 파일 경로 설정
    audio_file_path = os.path.join(upload_folder, 'test.wav')
    audio_file.save(audio_file_path)
    
    try:
        # 파일을 열 때 with 구문을 사용하여 자동으로 닫도록 합니다.
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            result = 'stt result: ' + transcript
            print(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # 에러 핸들링 추가

    return result

@app.route('/llm')
def home():
    words = ["저희", "농아인", "일반인", "화상", "회의", "서비스", "브릿지", "만들다", "12", "팀"]
    prompt = f"이 단어들을 자연스러운 한문장으로 만들어줘: {', '.join(words)}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": f"{prompt}"}
        ]
    )

    generated_text = response.choices[0].message.content
    generated_text = generated_text.split(':')[1][2:-3] # 생성된 문장만 추출
    return render_template('result.html', generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
