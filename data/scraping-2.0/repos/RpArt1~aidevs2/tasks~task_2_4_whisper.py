from tasks.abstractTask import AbstractTask
import re
from openai import OpenAI
import requests
import logging
from utils.config import Config
import io
class WhisperTask(AbstractTask):
    
    #documentation => https://platform.openai.com/docs/guides/speech-to-text/quickstart
    OpenAI_WHISPER_MODEL = "whisper-1"
    FILE_PATH = "/tmp/omfg.mp3"

    def solve_task(self):
        return super().solve_task()
    
    def url_from_task(self) -> str:
        try: 
            task_body = self.assignment_body['msg']
            url_pattern = re.compile(r'https?://\S+|www\.\S+')
            url = url_pattern.findall(task_body)[0]  
        except Exception as e:
            raise UrlParsingException(f"UrlParsingException: Cannot retreive url from task assigment: {str(e)} ")
        if (url == None or not isinstance(url, str)):
            raise UrlParsingException(f"UrlParsingException: url is as null as hell or not a string")
        return url
    
    def open_file(self, url :str) -> io.BufferedReader: 
        try: 
            response = requests.get(url)
            with open(self.FILE_PATH, 'wb') as file:
                file.write(response.content)
            audio_file = open(self.FILE_PATH, "rb")
        except OSError  as ose:
            raise FileProcessingException(f"File cannot be open: {ose}")
        except Exception as e: 
            raise FileProcessingException(f"File cannot be open: {e}")
        if (audio_file == None):
            raise FileProcessingException(f"Guess :)...")
        return audio_file

    def whisper_that_file(self, audio_file: io.BufferedReader) ->str:
        try:  
            if(self.mock):
                transcript = "nonononno DO NOT REDEEM"
            else:
                client = OpenAI(api_key=Config().open_api_key)
                ai_response = client.audio.transcriptions.create(
                    model=self.OpenAI_WHISPER_MODEL, 
                    file=audio_file
                )
                transcript = ai_response.text
                logging.info(f"Transcript is: \"{transcript}\"")
        except Exception as e:
            raise OpenAIException(f"File wasn't processed correctly by openai: {e}")
        if(transcript == None):
            raise OpenAIException("File was null")
        return transcript
    
    def process_task_details(self):
        try:
            url = self.url_from_task()
            audio_file = self.open_file(url)
            return self.whisper_that_file(audio_file)
          
        except UrlParsingException as upe:
            logging.error(upe)
            exit
        except FileProcessingException as fpe:
            logging.error(fpe)
            exit
        except OpenAIException as oaie:
            logging.error(fpe)
            exit
        except Exception as e:
            logging.error(f"Yeah some error, fuck you and see you tommorow: {e}")
            exit
      
class UrlParsingException(Exception):
    super        
class FileProcessingException(Exception):
    super      
class OpenAIException(Exception):
    super