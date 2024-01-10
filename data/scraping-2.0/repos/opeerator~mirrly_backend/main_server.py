#!/usr/bin/env python3
import os
import sys
import time
import random
import threading
import subprocess
import multiprocessing
import atexit
from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO, emit
from google.cloud import texttospeech
from google.oauth2 import service_account
import openai
import speech_recognition as sr
import cv2

from mode_config import ModeManager
from head_control import HeadMotors
from torso_control import TorsoMotors

mode_manager = ModeManager()
head_motors = HeadMotors()
torso_motors = TorsoMotors()
current_move = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'somesecret'
socketio = SocketIO(app)

current_move = None

eye_pos_checker = 0

os.environ['QT_QPA_PLATFORM'] = 'xcb'

openai.api_key = "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"

# Webcam indexes
webcam1_index = 0  # Index of the first webcam
webcam2_index = 2  # Index of the second webcam

# set up the TTS client
cr = service_account.Credentials.from_service_account_file(
    'socialrobotics-381420-7add1789cf22.json')
client = texttospeech.TextToSpeechClient(credentials=cr)
voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
)
audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.LINEAR16
)

idle_v = [
    "bored",
    "friend",
    "game",
    "hungry",
    "lab",
    "see",
    "shy",
    "stare",
    "weather"
]

is_speech_playing = False  # Variable to track if speech is currently playing

############################################
############################################
############################################

def chat_bot_module(text):
    prompt = text
    model = "text-davinci-002"
    temperature = 0.5
    max_tokens = 100

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    print(response.choices[0].text)
    return response.choices[0].text


def say(text):
    th = threading.Thread(target=generate_speech, args=(text, ))
    th.start()
    
def play_speech(name):
    global is_speech_playing
    
    while True:
        # Check if speech is already playing
        if is_speech_playing:
            return
        
        # Set is_speech_playing to True to indicate that speech is starting
        is_speech_playing = True
        
        # Amplify the sound using amixer
        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "300%"], capture_output=True)
        
        if name == "random":
            probability = 0.08
            if random.random() < probability:
                
                s_sound = random.choice(idle_v)
                subprocess.run(["aplay", f'tts/Daphne/{s_sound} (mp3cut.net).wav'], capture_output=True)
        else:
            # Play the amplified sound using aplay
            subprocess.run(["aplay", f'tts/Daphne/{name} (mp3cut.net).wav'], capture_output=True)
        
        # Set is_speech_playing to False to indicate that speech has finished playing
        is_speech_playing = False
        
        if name == "greeting":
            break
        
        time.sleep(0.5)


def generate_speech(text):
    # set up the TTS request
    synthesis_input = texttospeech.SynthesisInput(text=text)
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    # save the audio to a file
    random_number = random.randint(100, 999)
    with open(f'speech_{random_number}.wav', 'wb') as out:
        out.write(response.audio_content)

    # os.system(f'aplay speech_{random_number}.wav')
    
    # Amplify the sound using amixer
    subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "100%"], capture_output=True)

    # Play the amplified sound using aplay
    subprocess.run(["aplay", f'speech_{random_number}.wav'], capture_output=True)

    # Reset the sound amplification using amixer
    subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "100%"], capture_output=True)

    sys.exit()



def initial_positions():
    start_head_positions = head_motors.current_pos('all')

    """
    # Go to start positions - HEAD
    if start_head_positions[0] != int(807):
        head_motors.move("head_yaw", 807, 400)

    if start_head_positions[0] != int(119):
        head_motors.move("head_pitch", 119, 1000)

    if start_head_positions[0] != int(282):
        head_motors.move("eye_brow_l", 282, 500)

    if start_head_positions[0] != int(795):
        head_motors.move("eye_brow_r", 795, 500)

    if start_head_positions[0] != int(771):
        head_motors.move("eye_self", 771, 500)

    """
    

    # Go to start positions - HANDS
    torso_motors.arm_move('r_shoulder', 90, 0.02)  # Plate Top 90, Plate Front 0, Plate Down -90
    torso_motors.arm_move('l_shoulder', -90, 0.02)  # Plate Top -90, Plate Front 0, Plate Down 90
    torso_motors.arm_move('arm_r', -10, 0.02)  # -10 (Top) to 90 (Down)
    torso_motors.arm_move('arm_l', 85, 0.02)  # -20 (Down) to 85 (up)

    """
    torso_motors.arm_move('r_shoulder', 0, 0.02)  # Plate Top 90, Plate Front 0, Plate Down -90
    torso_motors.arm_move('l_shoulder', 0, 0.02)  # Plate Top -90, Plate Front 0, Plate Down 90
    torso_motors.arm_move('arm_r', 90, 0.02)  # -10 (Top) to 90 (Down)
    torso_motors.arm_move('arm_l', -20, 0.02)  # -20 (Down) to 85 (up)
    """

def eye_brow_idle():
    while True:
        probability = 0.07  # Probability of eye brow idle movement
        blink_speed = random.choice([400, 700, 1000])
        if random.random() < probability and eye_pos_checker < 800:
            try:
                head_motors.move("eye_brow_l", 282, blink_speed)
                time.sleep(0.01)
                head_motors.move("eye_brow_r", 795, blink_speed)
                if blink_speed == 1000:
                    time.sleep(0.2)
                elif blink_speed == 700:
                    time.sleep(0.5)
                elif blink_speed == 400:
                    time.sleep(0.7)
                else:
                    pass
                head_motors.move("eye_brow_l", 467, blink_speed)
                time.sleep(0.01)
                head_motors.move("eye_brow_r", 643, blink_speed)
            except SerialException:
                sleep(0.01)  # Maybe don't do this, or mess around with the interval
                continue
        time.sleep(0.5)

def eye_roll():
    while True:
        probability = 0.1  # Probability of eye movement
        eye_roll_speed = random.choice([500, 700, 1000])
        if random.random() < probability:
            random_value = random.randint(720, 919)
            eye_pos_checker = random_value
            head_motors.move("eye_self", random_value, eye_roll_speed)
        time.sleep(0.5)

def yaw_roll():
    while True:
        probability = 0.07  # Probability of eye movement
        if random.random() < probability:
            random_value = random.randint(590, 970)
            random_speed = random.randint(300, 500)
            head_motors.move("head_yaw", random_value, random_speed)
        time.sleep(0.5)

def pitch_roll():
    while True:
        probability = 0.07  # Probability of eye movement
        if random.random() < probability:
            random_value = random.randint(119, 200)
            random_speed = random.randint(900, 950)
            head_motors.move("head_pitch", random_value, random_speed)
        time.sleep(0.5)


def show_webcam1(cam_index):
    # Open the webcam
    cap = cv2.VideoCapture(cam_index)

    # Set the desired frame width and height
    frame_width = 320
    frame_height = 240

    # Set the frame size for the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    while True:
        # Read the current frame
        ret, frame = cap.read()

        # Resize the frame
        frame = cv2.resize(frame, (frame_width, frame_height))

        cv2.imshow(f'Right Eye - {cam_index}', frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
    
def get_camera(camera_index):
    camera = cv2.VideoCapture(camera_index)
    while True:
        success, frame = camera.read()
        if success:
            yield frame
        else:
            break
    camera.release()

def generate_frames(camera):
    for frame in camera:
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed_left_eye')
def video_feed_0():
    camera = get_camera(0)
    return Response(generate_frames(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
@app.route('/video_feed_right_eye')
def video_feed_1():
    camera2 = get_camera(2)
    return Response(generate_frames(camera2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
@app.route('/left_eye')
def left_eye_render():
    return render_template('video.html')
    
@app.route('/right_eye')
def right_eye_render():
    return render_template('video2.html')
    
def show_webcam_output(cam_index):
    # Create a face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Set the desired frame width and height
    frame_width = 320
    frame_height = 240

    # Open the webcam
    cap = cv2.VideoCapture(cam_index)  # Assuming webcam2 is used

    # Set the frame size for the webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Calculate the center of the frame
    frame_center_x = frame_width // 2

    # Calculate the range of yaw movement
    yaw_min = 590
    yaw_max = 970

    # Calculate the scaling factor
    scale_factor = (yaw_max - yaw_min) / frame_width

    # Initialize the previous face position
    prev_face_position = None
    is_moving = False

    while True:
        # Read the current frame
        ret, frame = cap.read()

        # Resize the frame
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Initialize the face center variable
        face_center_x = None

        if len(faces) > 0:
            # Select a random face from the detected faces
            selected_face = random.choice(faces)

            # Get the position of the selected face
            (x, y, w, h) = selected_face

            # Calculate the center of the selected face
            face_center_x = x + w // 2

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the horizontal movement based on the face position relative to the frame center
        if face_center_x is not None:
            horizontal_movement = frame_center_x - face_center_x

            # Calculate the yaw movement based on the scaling factor
            yaw_movement = (-1 * int(horizontal_movement * scale_factor)) / 7
            print(f"YawMovement: {yaw_movement}")

            # Perform yaw movement based on the horizontal face movement
            try:
                current_yaw = head_motors.current_pos("head_yaw")[0]
            except:
                print("dynamixel Problem on yaw")
                continue
            new_yaw = current_yaw + yaw_movement
            yaw_change = abs(current_yaw - new_yaw)
            print(f"newYaw: {new_yaw}")
            print(f"change: {yaw_change}")
            # Check if the new yaw is within the range
            if yaw_min <= new_yaw <= yaw_max and abs(yaw_change) > 5 and is_moving == False:
                print("MOVE")
                is_moving = True
                head_motors.move("head_yaw", new_yaw, 500)
                is_moving = False

        # Display the frame
        # cv2.imshow('Webcam', frame)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
    
#######################################
#######################################
#######################################

@socketio.event
def test_connect():
    print("Connected to mirrly")

@app.after_request
def add_header(response):
    """Adding header's for robot's response.

    Returns:
        json: Server's response
    """
    response.headers['Cache-Control'] = "no-cache, no-store, must-revalidate"
    return response


@app.route("/")
def index():
    """A sample page for 0.0.0.0

    Returns:
        html: An html page. 
    """
    return render_template('index.html')


@app.route("/run/<mode_name>", methods=['POST'])
def run(mode_name):
    # Run different modes.
    mode_manager.run(mode_name)
    response = {'message': f'{mode_name} running'}
    if mode_manager.should_redirect(mode_name):
        response['redirect'] = True
    return jsonify(response)

@app.route("/status")
def status_check():
    battery_level = 90 # Placeholder for battery status.
    if mode_manager.is_running():
        return jsonify({"status": "processing", f"battery_level":"{battery_level}"})
    else:
        return jsonify({"status": "idle", f"battery_level":"{battery_level}"})
    
# @app.route("/head/move/<component>/<gpos>", methods=['POST'])
@socketio.on('move/head')
def move_head_m(message):
    res = head_motors.move(message['component'], message['gpos'])
    return jsonify({"status": f"{res}"})
    
@socketio.on('move/body')
def body_movement(message):
    if message['direction'] == 'stop':
        res = torso_motors.release_motors(message['direction'])
    else:
        res = torso_motors.move(message['direction'], message['speed'])
    return jsonify({"status": f"{res}"})

@socketio.on('move/hand')
def body_movement(message):
    res = torso_motors.arm_move(message['comp'], message['angle'])
    return jsonify({"status": f"{res}"})

# @app.route("/head/getinfo/<component>", methods=['POST'])
@socketio.on('/head/getinfo/all')
def head_info(data):
    res = head_motors.current_pos(data['component'])
    emit('info', {'data': res})

@app.route("/stop/<mode_name>", methods=['POST'])
def stop(mode_name):
    # Turn off every system.
    mode_manager.stop(mode_name)
    return jsonify({'message': "Stopped"})
    

@socketio.on('move/random_control')
def random_control(message):
    global webcam1_process
    global webcam2_process
    global random_brow_process
    global random_eye_roll_process
    global random_pitch_roll_process
    global random_yaw_roll_process
    global random_speech_process
    
    if message['comp'] == 'cam0':
        if message['status'] == 'on':
            webcam1_process = multiprocessing.Process(target=show_webcam1, args=(webcam1_index,))
            webcam1_process.start()
        else:
            try:
                webcam1_process.terminate()
            except:
                pass
    elif message['comp'] == 'cam2':
        if message['status'] == 'on':
            webcam2_process = multiprocessing.Process(target=show_webcam_output, args=(webcam2_index,))
            webcam2_process.start()
        else:
            try:
                webcam2_process.terminate()
            except:
                pass
    elif message['comp'] == 'r_eye_brow':
        if message['status'] == 'on':
            random_brow_process = multiprocessing.Process(target=eye_brow_idle)
            random_brow_process.start()
        else:
            try:
                random_brow_process.terminate()
            except:
                pass
    elif message['comp'] == 'r_eye_roll':
        if message['status'] == 'on':
            random_eye_roll_process = multiprocessing.Process(target=eye_roll)
            random_eye_roll_process.start()
        else:
            try:
                random_eye_roll_process.terminate()
            except:
                pass
    elif message['comp'] == 'r_pitch_roll':
        if message['status'] == 'on':
            random_pitch_roll_process = multiprocessing.Process(target=pitch_roll)
            random_pitch_roll_process.start()
        else:
            try:
                random_pitch_roll_process.terminate()
            except:
                pass
    elif message['comp'] == 'r_yaw_roll':
        if message['status'] == 'on':
            random_yaw_roll_process = multiprocessing.Process(target=yaw_roll)
            random_yaw_roll_process.start()
        else:
            try:
                random_yaw_roll_process.terminate()
            except:
                pass
    elif message['comp'] == 'r_speech':
        if message['status'] == 'on':
            random_speech_process = multiprocessing.Process(target=play_speech, args=("random",))
            random_speech_process.start()
        else:
            try:
                random_speech_process.terminate()
            except:
                pass
    else:
        pass

# Create processes for each webcam
webcam1_process = None
webcam2_process = None

# Random actions processes            
random_brow_process = None
random_eye_roll_process = None
random_pitch_roll_process = None
random_yaw_roll_process = None
random_speech_process = None

if __name__ == "__main__":
    
    initial_positions()

    time.sleep(1)
    # Wake up motion
    
    head_motors.move("head_pitch", 276, 500)
    head_motors.move("eye_brow_l", 467, 400)
    head_motors.move("eye_brow_r", 643, 400)
    
    time.sleep(2)
    """
    torso_motors.hands_freq("arm_l", 50)
    torso_motors.hands_freq("arm_r", 50)
    torso_motors.arm_move('arm_l', 12) 
    torso_motors.arm_move('arm_r', 2.5) 
    head_motors.move("head_pitch", 110, 1000)
    time.sleep(2)
    torso_motors.arm_move('arm_r', 7.5) 
    time.sleep(0.5)
    torso_motors.hands_freq("arm_l", 1)
    torso_motors.hands_freq("arm_r", 1)
    multiprocessing.Process(target=play_speech, args=("greeting",)).start()
    
    torso_motors.hands_freq("l_shoulder", 50)
    torso_motors.hands_freq("arm_l", 50)
    # Wave Hands
    torso_motors.arm_move('l_shoulder', 12)  # 3-12.5
    time.sleep(0.5)
    torso_motors.arm_move('arm_l', 12)  # 6.5-12
    time.sleep(0.5)
    torso_motors.arm_move('arm_l', 9)  # 6.5-12
    time.sleep(0.5)
    torso_motors.arm_move('arm_l', 12)  # 6.5-12
    time.sleep(0.5)
    torso_motors.arm_move('arm_l', 9)  # 6.5-12
    time.sleep(0.5)
    torso_motors.arm_move('arm_l', 12)  # 6.5-12
    time.sleep(0.5)
    torso_motors.arm_move('arm_l', 9)  # 6.5-12
    time.sleep(0.5)
    torso_motors.arm_move('arm_l', 12)  # 6.5-12
    time.sleep(0.5)
    torso_motors.arm_move('arm_l', 9)  # 6.5-12
    time.sleep(0.5)
    torso_motors.arm_move('arm_l', 12)  # 6.5-12

    torso_motors.arm_move('l_shoulder', 11.5)  # 3-12.5
    torso_motors.arm_move('arm_l', 6.5)  # 6.5-12

    time.sleep(1)
    torso_motors.hands_freq("all", 50)
    time.sleep(0.5)
    torso_motors.release_hands("all")
    
    socketio.run(app, host='0.0.0.0', allow_unsafe_werkzeug=True)
    
    print("stopped")
    head_motors.disable_torque("all")
    torso_motors.release_motors()
    torso_motors.release_hands("all")
    
    try:
        webcam1_process.terminate()
        print("Webcam1 Process terminated.")
    except:
        print("Webcam1 termination failed/passed.")
        
    try:
        webcam2_process.terminate()
        print("Webcam2 Process terminated.")
    except:
        print("Webcam2 termination failed/passed.")

    try:
        random_brow_process.terminate()
        print("random_brow_process terminated.")
    except:
        print("random_brow_process termination failed/passed.")
        
    try:
        random_eye_roll_process.terminate()
        print("random_eye_roll_process terminated.")
    except:
        print("random_eye_roll_process termination failed/passed.")

    try:
        random_pitch_roll_process.terminate()
        print("random_pitch_roll_process terminated.")
    except:
        print("random_pitch_roll_process termination failed/passed.")
        
    try:
        random_yaw_roll_process.terminate()
        print("random_yaw_roll_process terminated.")
    except:
        print("random_yaw_roll_process termination failed/passed.")
        
    try:
        random_speech_process.terminate()
        print("random_speech_process terminated.")
    except:
        print("random_speech_process termination failed/passed.")
    """
