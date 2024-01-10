from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
# from pydub import AudioSegment
from openai import OpenAI
import os
import numpy as np
import cv2
import base64
import darknet
from time import time

app = Flask(__name__)

if __name__ == '__main__':
	app.run(debug=True, port=5000)

app = Flask(__name__)
CORS(app)
app.config['TEMPLATES_AUTO_RELOAD'] = True  # 템플릿 자동 리로드 활성화

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    # .env에 있는 OPENAI_API_KEY 가져옴 

# OpenAI GPT-3 API 엔드포인트와 API 키를 설정합니다.
# OPENAI_API_ENDPOINT = "https://api.openai.com/v1/completions"
client = OpenAI()
client.api_key = OPENAI_API_KEY

# 파일 저장 경로
upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
    
# yolo - 성현
def load_darknet_model(config_file, data_file, weights_file):
    network, class_names, class_colors = darknet.load_network(config_file, data_file, weights_file, batch_size=1)
    return network, class_names, class_colors
    
# 다크넷 컴파일
configPath="./darknet/cfg/yolov4-tiny-obj.cfg"
weightPath="./darknet/backup/yolov4-tiny-obj_best.weights"
metaPath="./darknet/data/obj.data"
network, class_names, class_colors = load_darknet_model(configPath, metaPath, weightPath)

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 동적으로 파일 경로 설정
    audio_file_path = os.path.join(upload_folder, 'test_audio.wav')
    audio_file.save(audio_file_path)
    
    try:
        # 파일을 열 때 with 구문을 사용하여 자동으로 닫도록 합니다.
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
            print(transcript)
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # 에러 핸들링 추가

    return transcript

def inference_on_video(video_path, network, class_names, threshold):
    cap = cv2.VideoCapture(video_path)
    detected_classes_list = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        darknet_width,darknet_height = 416, 416
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)

        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

        detections = darknet.detect_image(network, class_names, img_for_detect, thresh=threshold)

        detected_classes = get_detected_classes(detections)
        print("Detected Classes: {}".format(detected_classes))
        detected_classes_list.extend(detected_classes)

        darknet.free_image(img_for_detect)

    cap.release()
    return detected_classes_list

@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # 동적으로 파일 경로 설정 (원하는 파일 이름으로 변경)
    video_file_path = os.path.join(upload_folder, 'test_video.mp4')
    video_file.save(video_file_path)

    
    # threshold : 임계값 0~1
    try:
        detected_classes_list = inference_on_video(video_file_path, network, class_names, 0.7)
        result = request_chat_gpt(detected_classes_list)
        print(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # 에러 핸들링 추가

    return result #jsonify({'detected_classes': detected_classes_list})


# @app.route('/upload-video', methods=['POST'])
# def upload_video():
#     if 'video' not in request.files:
#         return jsonify({'error': 'No video file provided'}), 400

#     video_file = request.files['video']
#     if video_file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     # 동적으로 파일 경로 설정 (원하는 파일 이름으로 변경)
#     video_file_path = os.path.join(upload_folder, 'test_video.mp4')
#     video_file.save(video_file_path)
    
#     try:
#         # 욜로
#         text = inference(video_file_path)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500  # 에러 핸들링 추가
    
#     return text

# @app.route('/llm')
def request_chat_gpt(detected_classes):
    words = detected_classes #["저희", "농아인", "일반인", "화상", "회의", "서비스", "브릿지", "만들다", "12", "팀"]
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
    return generated_text

# Yolo

# @app.route('/process_image', methods=['POST'])
# def process_image():
#     data = request.get_json()  # JSON 데이터를 파싱
#     image_data = base64.b64decode(data['image'])
#     image = np.frombuffer(image_data, dtype=np.uint8)
#     image = cv2.imdecode(image, flags=cv2.IMREAD_COLOR)
    
#     # YOLO 모델을 사용하여 이미지 처리
#     results = performDetect(image=image)

#     # 감지된 객체의 정보를 반환
#     return jsonify(results)

# def convertBack(x, y, w, h):
#     # Convert from center coordinates to bounding box coordinates
#     xmin = int(round(x - (w / 2)))
#     xmax = int(round(x + (w / 2)))
#     ymin = int(round(y - (h / 2)))
#     ymax = int(round(y + (h / 2)))
#     return xmin, ymin, xmax, ymax

# def performDetect(image, thresh=0.25, configPath="../darknet/cfg/yolov4-tiny-obj.cfg", weightPath="../darknet/backup/yolov4-tiny-obj_best.weights", metaPath="../darknet/data/obj.data"):
#     # Load the network
#     network, class_names, class_colors = darknet.load_network(
#         configPath,
#         metaPath,
#         weightPath,
#         batch_size=1
#     )
    
#     # Resize the image and convert to RGB
#     width, height = darknet.network_width(network), darknet.network_height(network)
#     darknet_image = darknet.make_image(width, height, 3)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

#     # Copy the image data into the darknet image
#     darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())

#     # Perform the detection
#     detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
#     darknet.free_image(darknet_image)

#     # Convert detections to dictionary
#     detections_dict = []
#     for label, confidence, bbox in detections:
#         xmin, ymin, xmax, ymax = convertBack(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
#         detections_dict.append({
#             "label": label,
#             "confidence": float(confidence),
#             "bbox": [xmin, ymin, xmax, ymax]
#         })

#     return detections_dict


if __name__ == '__main__':
    app.run(debug=True)
