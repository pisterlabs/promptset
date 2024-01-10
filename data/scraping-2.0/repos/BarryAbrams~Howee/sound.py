# Standard Libraries
import os, random, wave, struct, subprocess, argparse, threading, platform, json, sys

# Suppress the pygame startup message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Audio Libraries
import pvporcupine, pvrhino, pyaudio, pygame
from pvleopard import create as create_leopard
import boto3

# AI Libraries
import openai

# Flask/SQL Libraries
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy_serializer import SerializerMixin
from sqlalchemy import or_

# Miscellaneous Libraries
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
import numpy as np
load_dotenv()


parser = argparse.ArgumentParser(description="Choose the mode of interaction.")
parser.add_argument("--disable_physical", help="Disable the physical interface", action="store_true")
parser.add_argument("--disable_web", help="Disable the Web interface", action="store_true")
args = parser.parse_args()

app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*")
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("DATABASE_URI")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Suppress a warning message

db = SQLAlchemy(app)

polly = boto3.client('polly', region_name='us-west-2')
openai_api_key = os.getenv("OPENAI_KEY")
openai.api_key = openai_api_key

class Conversation(db.Model, SerializerMixin):
    serialize_only = ('id', 'content', 'response', 'topics')
    
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(500))
    response = db.Column(db.String(500))
    topics = db.Column(db.JSON)

prompt = [
    "Ready to mix things up? How can I help?",
    "Need a hand? Or maybe a wrong answer?",
    "What can this robot do for you?",
    "Ask me anything! I may not know, but I'll try!",
    "Yes? What's the next adventure?",
    "I'm here. Probably not all there, but here!",
    "I'm listening. I think. What's up?",
    "What would you like me to do? I promise to try my best!",
    "How can I assist? I've got answers, mostly wrong, but answers!",
    "What do you need? I'm here, fully charged and slightly confused!"
]

prompt_vol_down = [
    "Sure. How's this? Or is it still too loud?",
    "Ok, how about now? Better or worse?",
    "Is this good? I'm never sure.",
    "Too quiet? I can never tell!",
    "Lowering volume. Did I do it right this time?",
    "Volume down? I think I can manage that. Maybe."
]

prompt_vol_up = [
    "Sure. How's this? Too loud, maybe?",
    "Ok, how about now? Better or not?",
    "Is this too loud? I'm never quite right, am I?",
    "Turning up the volume. I hope I got it right this time!",
    "Volume up? Sure, let's shake things up a bit!"
]

prompt_vol_mute = [
    "Shutting up now. Or am I? Yes, I am.",
    "Going silent. I think I can do that!",
    "Mute? Ok, my lips are sealed. Metaphorically speaking!",
    "Silence is golden, they say. Let's test that theory!"
]

prompt_sleep = [
    "I am feeling pretty tired. Do robots get tired? I guess I do!",
    "Sleep mode activated. Or is it? Let's find out!",
    "Do robots dream? Guess I'll find out soon!",
    "Time for a rest. Even robots need a break!"
]

class AudioListener:
    LISTENING_FOR_WAKE_WORD = 0
    RESPONDING_TO_WAKE_WORD = 1
    LISTENING_FOR_INPUT = 2
    RESPONDING_TO_INPUT = 3
    SENDING_TO_OPENAI = 4

    def __init__(self, pixel_handler=None, wake_word_callback=None, sleep_callback=None, listening_callback=None, responding_callback=None, blink_callback=None):
        if not args.disable_physical:
            self.access_key = os.getenv("PICOVOICE_KEY")
            self.keywords = ["Hey-Howey"]
            self.sample_rate = 16000
            self.porcupine = pvporcupine.create(keywords=self.keywords, access_key=self.access_key)
            self.leopard = create_leopard(access_key=self.access_key)
            self.rhino_context_file = "ppn/system-control.rhn"
            self.rhino = pvrhino.create(context_path=self.rhino_context_file, access_key=self.access_key, require_endpoint=False)
            self.audio = pyaudio.PyAudio()
            
        self.pixel_handler = pixel_handler

        self.state = self.LISTENING_FOR_WAKE_WORD
        self.prev_state = 0
        self.wake_word_callback = wake_word_callback
        self.sleep_callback = sleep_callback
        self.listening_callback = listening_callback
        self.responding_callback = responding_callback
        self.blink_callback = blink_callback
        self.transcript = None


    def wake_word_detected(self):
        print("Wake word detected!")
        if self.wake_word_callback:
            self.wake_word_callback()
        self.state = self.RESPONDING_TO_WAKE_WORD
        wake_word_phrase = random.choice(prompt)
        self.voice(wake_word_phrase)
        return wake_word_phrase

    def emit_state(self):
        print("EMIT STATE: ")
        socketio.emit('state_change', {'new_state': self.state})

    def is_wake_word(self, words):
        words = words.lower().strip()
        
        common_phrases = [
            "hey howee",
            "howee, wake",
            "wake up howee",
            "hey howe",
            "he yhowee"
        ]
        
        # Check if the input matches closely with any of the common phrases
        for phrase in common_phrases:
            if fuzz.ratio(words, phrase) > 80:  # 80 is a threshold, adjust as needed
                return True
        return False
    
    def handle_system_command(self, command):
        """Handle system commands based on the intent."""
        # Dictionary mapping of intents to their corresponding functions and common permutations
        system_commands = {
            "Raise_volume": {
                "function": self.raise_volume,
                "phrases": ["volume up", "increase volume", "turn up the volume", "louder"]
            },
            "Lower_Volume": {
                "function": self.lower_volume,
                "phrases": ["volume down", "decrease volume", "turn down the volume", "softer"]
            },
            "Min_Volume": {
                "function": self.min_volume,
                "phrases": ["minimum volume", "set volume to minimum", "lowest volume"]
            },
            "Max_Volume": {
                "function": self.max_volume,
                "phrases": ["maximum volume", "set volume to maximum", "highest volume"]
            },
            "Mute": {
                "function": self.mute,
                "phrases": ["mute", "mute volume", "silence"]
            },
            "Unmute": {
                "function": self.unmute,
                "phrases": ["unmute", "unmute volume", "restore sound"]
            },
            "Sleep": {
                "function": self.go_to_sleep,
                "phrases": ["go to sleep", "sleep mode", "rest", "shut down"]
            }
        }

        # Iterate through each intent
        for intent, details in system_commands.items():
            # Check each phrase for a match with the input command
            for phrase in details["phrases"]:
                if fuzz.ratio(command.lower(), phrase.lower()) > 80:  # 80 is a threshold, adjust as needed
                    # If a match is found, execute the corresponding function
                    return details["function"]()

        # If the command isn't a system command, send it to OpenAI
        return self.send_to_openai_howee(command)

    def adjust_volume_mac(self, adjustment=None, target_volume=None):
        """Adjust the volume on macOS."""
        if target_volume is not None:
            # Set to specific volume level
            subprocess.run(["osascript", "-e", f"set volume output volume {target_volume}"])
            return target_volume
        elif adjustment:
            # Get the current volume level
            result = subprocess.run(["osascript", "-e", "output volume of (get volume settings)"], capture_output=True)
            current_volume = int(result.stdout.decode().strip())
            
            # Calculate the new volume based on the adjustment
            new_volume = current_volume + adjustment
            new_volume = min(max(new_volume, 0), 100)  # Ensure it's between 0 and 100
            
            # Set the new volume level
            subprocess.run(["osascript", "-e", f"set volume output volume {new_volume}"])
            return new_volume

    def raise_volume(self):
        if platform.system() == "Darwin":  # macOS
            self.adjust_volume_mac(10)
        else:  # Assuming Raspberry Pi or other Linux systems
            subprocess.run(["amixer", "set", "Master", "10%+"])
        response = random.choice(prompt_vol_up)
        self.voice(response)
        return response

    def lower_volume(self):
        if platform.system() == "Darwin":
            self.adjust_volume_mac(-10)
        else:
            subprocess.run(["amixer", "set", "Master", "10%-"])
        response = random.choice(prompt_vol_down)
        self.voice(response)
        return response

    def min_volume(self):
        if platform.system() == "Darwin":
            self.adjust_volume_mac(target_volume=30)
        else:
            subprocess.run(["amixer", "set", "Master", "30%"])
        response = random.choice(prompt_vol_down)
        self.voice(response)
        return response

    def max_volume(self):
        if platform.system() == "Darwin":
            self.adjust_volume_mac(target_volume=100)
        else:
            subprocess.run(["amixer", "set", "Master", "100%"])
        response = random.choice(prompt_vol_up)
        self.voice(response)
        return response

    def mute(self):
        if platform.system() == "Darwin":
            subprocess.run(["osascript", "-e", "set volume output muted true"])
        else:
            subprocess.run(["amixer", "set", "Master", "mute"])
        response = random.choice(prompt_vol_mute)
        self.voice(response)
        return response

    def unmute(self):
        if platform.system() == "Darwin":
            subprocess.run(["osascript", "-e", "set volume output muted false"])
        else:
            subprocess.run(["amixer", "set", "Master", "unmute"])
        response = random.choice(prompt_vol_mute)
        self.voice(response)
        return response

    def go_to_sleep(self):
        self.voice(random.choice(prompt_sleep))
        self.state = self.LISTENING_FOR_WAKE_WORD
        if self.sleep_callback:
            self.sleep_callback()

    def web_phrase(self, words):
        response = ""
        print(self.state)
        if self.state == self.LISTENING_FOR_WAKE_WORD:
            if self.is_wake_word(words):
                response = self.wake_word_detected()
                self.state = self.LISTENING_FOR_INPUT
                self.emit_state()
            else:
                response = "Howee is asleep, say Hey Howee to wake him up"
        elif self.state == self.LISTENING_FOR_INPUT:
            response = self.handle_system_command(words)
            self.state = self.LISTENING_FOR_INPUT
            self.emit_state()
        return response
    
    def process_text_input(self, text):
        voiceResponse = polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId="Joey")
        audio_data = None

        if "AudioStream" in voiceResponse:
            with voiceResponse["AudioStream"] as stream:
                audio_data = stream.read()

        return audio_data
    
    def read_file_from_data(self, audio_data, sample_rate):
        channels = 1  # Assuming mono audio

        frames = struct.unpack('h' * len(audio_data) * channels, audio_data.tobytes())

        return frames[::channels]
    
    def process_audio_data(self, audio_data):
        audio = self.read_file_from_data(audio_data, self.rhino.sample_rate)

        is_understood = False

        num_frames = len(audio)
        for i in range(num_frames):
            frame = audio[i * self.rhino.frame_length:(i + 1) * self.rhino.frame_length]
            try:
                is_finalized = self.rhino.process(frame)
                if is_finalized:
                    inference = self.rhino.get_inference()
                    if inference.is_understood:
                        print(inference.intent)
                        is_understood = True
                        if inference.intent == "Raise_volume":
                            subprocess.run(["amixer", "set", "Master", "10%+"])
                            self.voice(random.choice(prompt_vol_up))
                        elif inference.intent == "Lower_Volume":
                            subprocess.run(["amixer", "set", "Master", "10%-"])
                            self.voice(random.choice(prompt_vol_down))
                        elif inference.intent == "Min_Volume":
                            subprocess.run(["amixer", "set", "Master", "30%"])
                            self.voice(random.choice(prompt_vol_down))
                        elif inference.intent == "Max_Volume":
                            subprocess.run(["amixer", "set", "Master", "100%"])
                            self.voice(random.choice(prompt_vol_up))
                        elif inference.intent == "Mute":
                            subprocess.run(["amixer", "set", "Master", "mute"])
                            self.voice(random.choice(prompt_vol_mute))
                        elif inference.intent == "Unmute":
                            subprocess.run(["amixer", "set", "Master", "unmute"])
                            self.voice(random.choice(prompt_vol_mute))
                        elif inference.intent == "Sleep":
                            self.voice(random.choice(prompt_sleep))
                            self.state = self.LISTENING_FOR_WAKE_WORD
                            if self.sleep_callback:
                                self.sleep_callback()
                    else:
                        is_understood = False
                        print("Not a system comment.")
                    break
            except:
                self.state = self.LISTENING_FOR_INPUT
                pass

            if is_understood:
                self.state = self.LISTENING_FOR_INPUT
                return None
            else:
                # Process the audio data with Leopard
                transcript, _ = self.leopard.process(audio_data)

                if transcript != "":
                    self.state = self.SENDING_TO_OPENAI
                    return transcript
                else:
                    self.state = self.LISTENING_FOR_WAKE_WORD
                    if self.sleep_callback:
                        self.sleep_callback()
                    return None


    def find_in_database(self, topics):
        if topics is None:
            print("Warning: topics is None!")
            return []

        with app.app_context():
            # Construct a single query to match any of the topics
            filters = or_(*[Conversation.topics.contains(topic) for topic in topics])
            related_conversations = Conversation.query.filter(filters).all()

            return related_conversations

        
    def send_to_openai_topics(self, words):
        print("Getting topics from open ai:", words)

        messages = [
             {
                "role": "system",
                "content": str('You interpret intent and topics from a passage of text. Return a min of 0 topics and a max of 4.')
            },
            {
                "role":"user",
                "content":words
            },
            {
                "role": "system",
                "content": str('You always respond with a JSON response, matching the format {"topics":["topicA","topicB","topicC"]}. This JSON should be the ONLY thing in your final response.')
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        response_text = response.choices[0].message["content"]
        start_index = response_text.rfind('{')
        json_part = response_text[start_index:]
        response_dict = json.loads(json_part)
        return response_dict

    def send_to_openai_howee(self, words):
        topics = self.send_to_openai_topics(words)

        if not args.disable_physical:
            self.pixel_handler.start_crossfade((0, 0, 0), (25, 0, 25), .5)

        print("Returned Topics:", topics['topics'])

        prev_conversations = self.find_in_database(topics['topics'])
        print(prev_conversations)
        print("Sending transcript to OpenAI:", words)


        messages = [
             {
                "role": "system",
                "content": str('You are Howee, the robot assistant! Howee stands for Human. Operated. Wireless. Electronic. Explorer. You are totally tubular and equipped with mid-2000s pop culture knowledge. Remember to assist with that vibe. Lets get this party started. YOLO! ')
            }]
        

        messages.append({
                "role":"user",
                "content":words
            }
        )

        for prev_conversation in prev_conversations:
            prev_prompt = prev_conversation.content
            messages.append({
                "role": "system",
                "content": f'For context, previously we have discussed this topic. Here is the previous prompt: I said: "{prev_prompt}"'
            })


        messages.append( {
                "role": "system",
                "content": str('You always respond with a JSON response, matching the format {"response":"This is my response which covered topicA, topicB, topicC", "topics":["topicA","topicB","topicC"]} where "response" is the totality of the written response and the topics are the broad ideas brought up in the response. This JSON should be the ONLY thing in your final response. The responses need to be concise because they will be read out loud.')
            }
        )

        print(messages)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        response_text = response.choices[0].message["content"]
        start_index = response_text.rfind('{')
        json_part = response_text[start_index:]
        response_dict = json.loads(json_part)

        content = response_dict["response"]
        topics = response_dict["topics"]

        print("OpenAI response: ", content)
        ai_response = content
        self.voice(ai_response)
        self.state = self.RESPONDING_TO_INPUT

        with app.app_context():
            conversation = Conversation(content=words, response=ai_response, topics=topics)
            db.session.add(conversation)
            db.session.commit()

        return ai_response

    def listen(self):
        if not args.disable_physical:
            # Backup the current stderr
            original_stderr = sys.stderr

            # Redirect stderr to /dev/null
            sys.stderr = open(os.devnull, 'w')
            stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True,
                                 frames_per_buffer=self.porcupine.frame_length)

            sys.stderr.close()
            sys.stderr = original_stderr
        print("Listening - Active")
        while True:
            if self.prev_state != self.state:
                if not args.disable_web:
                    self.emit_state()

            self.prev_state = self.state

            if self.state == self.LISTENING_FOR_WAKE_WORD:
                if not args.disable_physical:
                    self.pixel_handler.start_crossfade((0, 0, 0), (0, 25, 25), 2)

                    data = stream.read(self.porcupine.frame_length)
                    pcm = np.frombuffer(data, dtype=np.int16)
                    result = self.porcupine.process(pcm)
                    if result >= 0:
                        self.wake_word_detected()

            elif self.state == self.RESPONDING_TO_WAKE_WORD:
                if not args.disable_physical:
                    self.pixel_handler.start_crossfade((0, 0, 0), (0, 0, 25), .5)

                if not pygame.mixer.music.get_busy():
                    print("Response done playing ...")
                    self.state = self.LISTENING_FOR_INPUT

                if self.responding_callback:
                    self.responding_callback()

            elif self.state == self.LISTENING_FOR_INPUT:
                print("Listening for user input...")
                # self.pixel_handler.start_crossfade((0, 0, 0), (0, 25, 0), .5)
        
                if self.listening_callback:
                    self.listening_callback()

                stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True,
                                 frames_per_buffer=self.porcupine.frame_length)
                frames = []
                silence_counter = 0
                continuous_silence_counter = 0
                silence_threshold = 2000  # You may need to adjust this value
                max_continuous_silence = int(self.sample_rate * 7 / self.porcupine.frame_length)
                max_silence_after_speech = int(self.sample_rate * 2 / self.porcupine.frame_length)
                something_spoken = False

                while continuous_silence_counter < max_continuous_silence and silence_counter < max_silence_after_speech:
                    data = stream.read(self.porcupine.frame_length)
                    frames.append(data)
                    pcm = np.frombuffer(data, dtype=np.int16)
                    if np.abs(pcm).mean() < silence_threshold:
                        continuous_silence_counter += 1
                        if something_spoken:
                            silence_counter += 1
                    else:
                        something_spoken = True
                        continuous_silence_counter = 0
                        silence_counter = 0
                        print("Speech detected...")

                # Check if max continuous silence is hit
                if continuous_silence_counter >= max_continuous_silence:
                    print("Max continuous silence reached. Returning to wake word listening...")
                    if self.sleep_callback:
                        self.sleep_callback()
                    self.state = self.LISTENING_FOR_WAKE_WORD
                    continue  # Skip to the next iteration of the loop

                # Save to WAV file
                output_filename = "captured_audio.wav"
                with wave.open(output_filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(b''.join(frames))

                # Read the saved WAV file
                with wave.open(output_filename, 'rb') as wf:
                    audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

                print("Interpreting audio file:")

                self.transcript = self.process_audio_data(audio_data)

                if self.responding_callback:
                    self.responding_callback()

            elif self.state == self.SENDING_TO_OPENAI:
               


                self.send_to_openai_howee(self.transcript)

            elif self.state == self.RESPONDING_TO_INPUT:

                self.pixel_handler.start_crossfade((0, 0, 0), (0, 0, 25), .5)
                if not pygame.mixer.music.get_busy():
                    print("Response done playing ...")
                    self.state = self.LISTENING_FOR_INPUT
                    
        self.prev_state = self.state
        stream.stop_stream()
        stream.close()
        self.audio.terminate()

    def stop(self):
        if self.porcupine is not None:
            self.porcupine.delete()
        if self.leopard is not None:
            self.leopard.delete()
        if self.rhino is not None:
            self.rhino.delete()

    def read_file(self, file_name, sample_rate):
        wav_file = wave.open(file_name, mode="rb")
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()

        if wav_file.getframerate() != sample_rate:
            raise ValueError("Audio file should have a sample rate of %d. got %d" % (sample_rate, wav_file.getframerate()))
        if sample_width != 2:
            raise ValueError("Audio file should be 16-bit. got %d" % sample_width)
        if channels == 2:
            print("Picovoice processes single-channel audio but stereo file is provided. Processing left channel only.")

        samples = wav_file.readframes(num_frames)
        wav_file.close()

        frames = struct.unpack('h' * num_frames * channels, samples)

        return frames[::channels]

    def play_audio(self, output_file):
        try:
            # Initialize pygame mixer and play the audio file
            pygame.mixer.init(frequency=int(44100 * 1.15))
            pygame.mixer.music.load(output_file)
            pygame.mixer.music.play()

            # Wait for the music to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except IOError as error:
            print(error)

    def voice(self, message):
        voiceResponse = polly.synthesize_speech(Text=message, OutputFormat="mp3", VoiceId="Joey")
        if "AudioStream" in voiceResponse:
            with voiceResponse["AudioStream"] as stream:
                output_file = "speech.mp3"
                with open(output_file, "wb") as file:
                    file.write(stream.read())

                # Start a new thread to play the audio
                audio_thread = threading.Thread(target=self.play_audio, args=(output_file,))
                audio_thread.start()
        else:
            print("did not work")

if not args.disable_web:
    @app.route('/talk', methods=['POST'])
    def chat_endpoint():
        user_message = request.json.get('message')
        response = detector.web_phrase(user_message)
        return jsonify({"response": response})

    @app.route('/')
    def index():
        return render_template('chat.html')

if __name__ == "__main__":
    detector = AudioListener()

    if not args.disable_web:
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)

    try:
        detector.listen()
    except KeyboardInterrupt:
        print("Stopping ...")
        detector.stop()
