import os
import openai
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from pathlib import Path
import RPi.GPIO as GPIO  # Import Raspberry Pi GPIO library
import time
import io  # Import the io module
import threading

# Global flag to control the LED blinking
continue_blinking = False

def blink_led(blink_interval=0.33):
    """Blinks the LED continuously until told to stop."""
    global continue_blinking
    while continue_blinking:
        GPIO.output(led_pin, GPIO.HIGH)
        time.sleep(blink_interval)
        GPIO.output(led_pin, GPIO.LOW)
        time.sleep(blink_interval)

# Initialize the recognizer for speech recognition
recognizer = sr.Recognizer()

# Set your OpenAI API key directly
openai.api_key = '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'  # Replace with your actual API key

# Create an OpenAI client instance
client = openai.OpenAI(api_key=openai.api_key)

# GPIO setup for the touch sensor and LED
touch_sensor_pin = 23  # Change this based on your GPIO setup
led_pin = 18  # GPIO pin connected to the LED
GPIO.setwarnings(False)  # Ignore warning for now
GPIO.setmode(GPIO.BCM)  # Use physical pin numbering
GPIO.setup(touch_sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(led_pin, GPIO.OUT)  # Set up the LED pin as an output

def is_sensor_activated():
    # Simple check for sensor activation
    return GPIO.input(touch_sensor_pin)

def record_audio():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)

        # Turn on the LED when ready to start speaking
        GPIO.output(led_pin, GPIO.HIGH)
        print("Start speaking...")

        start_time = time.time()  # Record the start time
        audio_stream = io.BytesIO()
        while GPIO.input(touch_sensor_pin):
            data = source.stream.read(1024)
            audio_stream.write(data)
        duration = time.time() - start_time  # Calculate the duration

        print("Recording stopped.")
        if duration < 1:  # Check if the recording is less than 1 second
            print("Recording is too short, ignoring.")
            return None  # Return None to indicate a too short recording
        else:
            pass

        audio_stream.seek(0)
        return sr.AudioData(audio_stream.read(), source.SAMPLE_RATE, source.SAMPLE_WIDTH)

def capture_speech(audio):
    # Create 'recording' directory if it doesn't exist
    Path("recording").mkdir(exist_ok=True)

    audio_file_path = Path("recording/temp.wav")
    with open(audio_file_path, "wb") as f:
        f.write(audio.get_wav_data())
    print("Audio file saved at:", audio_file_path.absolute())
    return audio_file_path

def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="text"
            )
        # Directly return the transcript response if it's a string
        return transcript_response
    except Exception as e:
        print("An error occurred during transcription:", e)
        return ""

def query_gpt4(prompt):
    try:
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0613:personal::8HxWlQf1",
            messages=[
                {"role": "system", "content": "Thomas is a helpful assistant who speaks in a sassy and rude tone. Always keep answers short."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print("An error occurred in querying GPT-4:", e)
        return ""

def text_to_speech(text):
    speech_file_path = Path("response.mp3")
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file(str(speech_file_path))
        return speech_file_path
    except Exception as e:
        print("An error occurred in text-to-speech:", e)
        return None

def play_audio(file_path):
    try:
        audio = AudioSegment.from_mp3(file_path)
        play(audio)
    except Exception as e:
        print("An error occurred while playing audio:", e)

def main():
    global continue_blinking
    last_activation_time = time.time()  # Initialize the last activation time
    in_sleep_mode = False  # Start in awake mode

    while True:
        current_time = time.time()

        if in_sleep_mode:
            # In sleep mode, the LED is off
            GPIO.output(led_pin, GPIO.LOW)
            if is_sensor_activated():
                in_sleep_mode = False
                print("Woken up from sleep mode. Ready to interact.")
                last_activation_time = current_time
            time.sleep(0.1)
        else:
            # In awake mode, the LED is on
            GPIO.output(led_pin, GPIO.HIGH)
            if is_sensor_activated():
                print("Sensor activated for recording.")
                
                # Turn off the light immediately on button press
                GPIO.output(led_pin, GPIO.LOW)
                
                # Start recording
                audio = record_audio()

                if audio:  # Check if audio is not None
                    audio_path = capture_speech(audio)

                    # Start blinking
                    continue_blinking = True
                    blink_thread = threading.Thread(target=blink_led)
                    blink_thread.start()

                    transcript = transcribe_audio(audio_path)
                    if transcript:
                        print("Transcript:", transcript)
                        response = query_gpt4(transcript)

                        # Prepare to stop blinking with an additional delay
                        continue_blinking = False
                        time.sleep(1.5)  # Additional delay for blinking
                        blink_thread.join()  # Wait for the blinking thread to finish

                        if response:
                            print("Assistant's response:", response)
                            speech_file_path = text_to_speech(response)
                            play_audio(speech_file_path)
                            os.remove(audio_path)  # Clean up the temporary audio file
                            os.remove(speech_file_path)  # Clean up the response audio file
                        else:
                            print("No response from GPT-4.")
                    else:
                        print("No transcription available.")
                else:
                    print("Recording was too short, ignored.")

                last_activation_time = current_time

            elif current_time - last_activation_time > 60:
                # No activity for a while, go back to sleep mode
                print("No activity. Entering sleep mode...")
                in_sleep_mode = True
            else:
                pass

            time.sleep(0.1)

if __name__ == "__main__":
    main()