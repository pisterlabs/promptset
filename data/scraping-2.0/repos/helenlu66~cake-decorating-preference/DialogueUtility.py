import os
from pathlib import Path
from ConfigUtil import get_args, load_experiment_config
from AudioRecorder import AudioRecorder
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play


class DialogueUtility:
    def __init__(self, user_name='user', api_key=os.environ['OPENAI_API_KEY'] if 'OPENAI_API_KEY' in os.environ else None) -> None:
        self.client = OpenAI(api_key=api_key)
        self.user_name = user_name # the name of the human user
        self.speech_to_text_path = 'speech2text'  # recording of human user's speech, need to use string relative path
        self.text_to_speech_path = Path(__file__).parent /'text2speech'  # audio file of the robot's speech for the human user
        self.exp_config = load_experiment_config('experiment_config.yaml')
        self.speech_turn_num = 0 # the number of turns the robot has taken to speak
        self.listen_turn_num = 0 # the number of turns the human has taken to speak / the robot has taken to listen
        self.recorder = AudioRecorder(record_len=self.exp_config['human_speech_record_len'])

    def text_to_speech(self, text):
        """Call openai's text to speech to produce an audio file and play the audio file

        Args:
            text (string): the text to be turned into speech
        """
        # call openai's text to speech
        response = self.client.audio.speech.create(
            model="tts-1",
            response_format='mp3',
            voice="alloy",
            input=text
        )
        speech_file_path = self.text_to_speech_path  /  (self.user_name + str(self.speech_turn_num) + '.wav')
        response.stream_to_file(speech_file_path)
        audio_data = AudioSegment.from_file(speech_file_path)
        # Play the audio
        play(audio_data)
        self.speech_turn_num += 1        

    def record_human_speech(self, wait_len:float):
        """Record the human user's speech and save to human user speech2text file path
        """
        filepath = self.speech_to_text_path + '/' + (self.user_name + str(self.listen_turn_num) + '.wav')
        self.recorder.record_human_speech(filepath=filepath, wait_len=wait_len)
        return filepath
    
    def speech_to_text(self, filepath):
        """Open the audio file and send it to openai's speech to text to turn it into text
        """
        audio_file= open(filepath, "rb")
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format='text',
            language='en'
        )
        self.listen_turn_num += 1
        return transcript
    
# can run the following for testing
if __name__=="__main__":
    exp_config = load_experiment_config('experiment_config.yaml')
    args = get_args()
    dialogue_util = DialogueUtility(user_name=exp_config['user_name'], api_key=args.api_key if args.api_key else os.environ['OPENAI_API_KEY'])
    dialogue_util.text_to_speech(text=f"Hi {exp_config['user_name']}, where should I place the first candle?")
    human_speech_filepath = dialogue_util.record_human_speech()
    human_speech_text = dialogue_util.speech_to_text(human_speech_filepath)
    print(human_speech_text)



    