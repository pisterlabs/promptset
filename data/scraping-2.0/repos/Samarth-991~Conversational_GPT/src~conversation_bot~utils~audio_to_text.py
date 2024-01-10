from tenacity import retry, wait_random
import openai
import torch as th
import whisper
from whisper.audio import SAMPLE_RATE
import logging
import json

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


class AudioProcess:
    def __init__(self, model_name='small', device='cuda'):
        self.device = device
        if not th.cuda.is_available():
            logging.warning('Cuda Not available, using Device as CPU')
            self.device = 'cpu'
        self.model = whisper.load_model(model_name, device=self.device)  # in config
        print("{} whisper model loaded in {}".format(model_name,self.device))

    def get_info(self, audio, duration=30):
        clip_audio = whisper.pad_or_trim(audio, length=SAMPLE_RATE * duration)
        result = self.model.transcribe(clip_audio)
        info = self.get_conversation_info(result['text'])
        return result['language'],info

    def get_conversation_info(self, text_data):
        summary_template = """
                    Given the conversation {information} between two persons identify the Customer name and 
                    relationship manager name. Usually relationship manager asks customer if he is looking for
                    services or property and customer replies either yes or no.
                    Answer in JSON format with keys customer_name and representative_name only.If not sure mention ""
                    """
        summary_prompt_template = PromptTemplate(template=summary_template, input_variables=["information"])
        llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")

        chain = LLMChain(llm=llm, prompt=summary_prompt_template)
        answer = chain.run(information=text_data)
        try:
            answer = json.loads(answer)
        except EncodingWarning as err:
            logging.error("Open AI send multiple answers")
            answer = {"customer_name":"","representative_name":""}
        return answer

    def speech_to_text(self, audio_path):
        try:
            audio = whisper.load_audio(audio_path)
        except IOError as err:
            logging.error("{}".format(err))
            raise "Issue in loading audio file. If path is correct try sudo apt-get install ffmpeg"
        audio_language, conv_info = self.get_info(audio)
        if audio_language == 'en':
            res = self.model.transcribe(audio)
        else:
            res = self.model.transcribe(audio)
            res['text'] = self.translate_text(res['text'], orginal_text=audio_language, convert_to='English')
        audio_duration = audio.shape[0] / SAMPLE_RATE
        return res['text'], audio_duration, audio_language, conv_info

    @retry(wait=wait_random(min=5, max=10))
    def translate_text(self, text, orginal_text='ar', convert_to='en'):
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
