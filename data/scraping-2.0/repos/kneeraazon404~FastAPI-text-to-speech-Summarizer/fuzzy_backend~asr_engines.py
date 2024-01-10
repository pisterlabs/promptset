import os

# import torch
import array
import numpy
import json
import azure.cognitiveservices.speech as speechsdk
import openai

subscription_key = os.getenv("SPEECH_KEY")
subscription_key = "598d3431577d4431bf8ff02c77eb27b7"
service_region = "eastasia"
custom_endpoint = "1bd6d651-d0d9-41bf-b737-f4463cc2eca7"
openai.api_key = os.getenv("OPENAI_API_KEY")
gpt_model = os.getenv("GPT_MODEL")
language_code = "zh-HK"


def get_floatdata(frame):
    try:
        short_array = array.array("h", frame)
    except (TypeError, ValueError):
        short_array = []
    # short_array = array.array('h', frame)
    float_array = []

    for sample in short_array:
        if sample < 0:
            float_array.append(float(sample / 32768.0))
        else:
            float_array.append(float(sample / 32767.0))

    return float_array


def mean_energy(frame):
    return numpy.sum(frame**2) / numpy.float64(len(frame))


def save_wave(float_data, sample_rate):
    import struct, wave

    rec_filename = "soundfile.wav"
    raw_floats = [x for x in float_data]
    floats = array.array("f", raw_floats)
    samples = [int(sample * 32767) for sample in floats]
    raw_ints = struct.pack("<%dh" % len(samples), *samples)

    wf = wave.open(rec_filename, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(raw_ints)
    wf.close()


class ASREngine:
    def __init__(self, lang):
        self.lang = lang

    def get_engine(self):
        pass

    def get_transcription(self):
        pass


class AzureEngine(ASREngine):
    def __init__(self, language_code):
        super().__init__(language_code)
        self.partial_res = ""
        self.text_res = ""
        # Configure speech recognition
        speech_config = speechsdk.SpeechConfig(
            subscription=subscription_key, region=service_region
        )
        speech_config.endpoint_id = custom_endpoint
        speech_config.speech_recognition_language = language_code
        speech_config.enable_streaming = True

        self.audio_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=self.audio_stream)

        # Create a speech recognizer with streaming support
        self.speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        # Connect the callbacks to the speech recognizer events
        self.speech_recognizer.recognizing.connect(self.process_audio_chunk)
        self.speech_recognizer.recognized.connect(self.process_audio_chunk)
        self.speech_recognizer.canceled.connect(self.process_audio_chunk)
        self.speech_recognizer.start_continuous_recognition_async()

        self.audio_frames = None
        self.audio_byte_num = 0

    # Set up the callbacks for handling streaming data
    def process_audio_chunk(self, args):
        result = args.result
        if result.reason == speechsdk.ResultReason.RecognizingSpeech:
            print("Recognizing:", result.text)
            self.partial_res = result.text
        elif result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print("Recognized: {}".format(result.text))
            self.text_res = result.text
            self.partial_res = ""
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech recognition canceled. Reason:", cancellation_details.reason)
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details:", cancellation_details.error_details)

    def get_transcription(self, frame):
        # get float data
        # self.speech_recognizer.stop_continuous_recognition()
        # print("transcription was done.")
        if self.audio_frames is None:
            self.audio_frames = frame
        else:
            self.audio_frames += frame

        if len(self.audio_frames) > 2048:
            self.audio_stream.write(self.audio_frames)
            self.audio_frames = None

        stt_res = json.dumps({"partial": self.partial_res})
        if self.text_res:
            stt_res = json.dumps({"text": self.text_res})
            self.text_res = ""

        # check if we will get transcription
        print("stt_res: ", stt_res)
        return stt_res
