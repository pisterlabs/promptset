import os
import torch as th
import whisper
from whisper.audio import SAMPLE_RATE
import logging
from tenacity import retry, wait_random
import openai
import requests
import time
# os.environ['OPENAI_API_KEY'] = "sk-<API KEY>"

class WHISPERModel:
    def __init__(self, model_name='small', device='cuda',openai_flag=False):
        self.logger = self.create_logger()
        self.device = device
        self.model_latency = 0
        self.openai_flag = openai_flag
        if not th.cuda.is_available():
            logging.warning('Cuda is not available, using Device as CPU.')
            self.device = 'cpu'
        if  model_name =='medium' :
            logging.warning("With higher model complexity conversion time will increase.")

        self.logger.info("Loading Model || Model size:{}".format(model_name))
        self.model = whisper.load_model(model_name, device=self.device)

    def get_info(self, audio_data, conv_duration=30):
        clip_audio = whisper.pad_or_trim(audio_data, length=SAMPLE_RATE * conv_duration)
        result = self.model.transcribe(clip_audio)
        return result['language']

    def speech_to_text(self, audio_path):
        self.logger.info("Reading url {}".format(audio_path))
        text_data = dict()
        audio_duration = 0
        conv_language = ""
        r = requests.get(audio_path)
        if r.status_code == 200:
            try:
                audio = whisper.load_audio(audio_path)
                conv_language = self.get_info(audio)
                stime = time.time()
                if conv_language !='en':
                    res = self.model.transcribe(audio,task='translate')
                    if self.openai_flag:
                        res['text'] = self.translate_text(res['text'], orginal_text=conv_language, convert_to='English')
                else:
                    res = self.model.transcribe(audio)
                self.model_latency = round(time.time()-stime)
                text_data['text'] = res['text']
                audio_duration = audio.shape[0] / SAMPLE_RATE
                logging.info("audio of {} conversion  took {}".format(audio_duration,self.model_latency))
            except IOError as err:
                logging.error("{}".format(err))
                raise f"Issue in loading audio {audio_path}"
        else:
            logging.error("Unable to reach for URL {}".format(audio_path))
        return text_data



    @retry(wait=wait_random(min=5, max=10))
    def translate_text(self, text, orginal_text='ar', convert_to='english'):
        prompt = f'Translate the following {orginal_text} text to {convert_to}:\n\n{orginal_text}: ' + text + '\n{convert_to}:'
        # Generate response using ChatGPT
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7
        )
        # Extract the translated English text from the response
        translation = response.choices[0].text.strip()
        return translation

    @staticmethod
    def create_logger():
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:- %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger = logging.getLogger("APT_Realignment")
        logger.setLevel(logging.INFO)

        if not logger.hasHandlers():
            logger.addHandler(console_handler)
        logger.propagate = False
        return logger

if __name__ == '__main__':
    url = "https://prypto-api.aswat.co/surveillance/recordings/5f53c28b-3504-4b8b-9db5-0c8b69a96233.mp3"
    audio2text = WHISPERModel()
    text = audio2text.speech_to_text(url)
