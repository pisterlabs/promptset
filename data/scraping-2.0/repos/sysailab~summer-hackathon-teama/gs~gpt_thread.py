from work_queue import WorkQueue
from model_controller import ModelController
from model_event import ModelEvent
import threading
import time 
import openai
from config import *

class GptThread:
    def __init__(self) -> None:
        openai.api_key = API_KEY
        
    
    def work(self):
        print("GPT Thread Started")
        while True:
            # TODO: GPT Queue를 모니터링 하고, 값이 들어오면 GPT API를 이용해 데이터 분류
            #       성공 시, Thread Model에게 분류한 데이터를 전달 후 event set
            #       실패 시, Thread Model에게 논문 추가 데이터 요청 후 event set
            datas = WorkQueue.gptQueue.get()
            texts: list = datas[0]
            event: ModelEvent = datas[1]

            for t in texts:
                print('=======================================')
                print(t)

            # TODO: GPT API를 통해 데이터 분류
            
            #title, authors, acknowledgements = self.get_paper_details(texts)
            
            #print(f'title : {title}\nauthors : {authors}\nacknowledements : {acknowledgements}')

            #################################
            time.sleep(3)
            event.set(ModelController.SIGNAL)
            
    def get_paper_details(self, text):
        response = openai.Completion.create(
            engine="text-davinci-003", # 현재 가장 성능이 좋은 모델 사용
            prompt=f"{text[0]}\{text[0]}\n{text[0]}n\nTitle of the paper:",
            temperature=0.3,
            max_tokens=60
        )
        title = response.choices[0].text.strip()

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"{text[0]}\{text[0]}\n{text[0]}\n\nAuthors of the paper:",
            temperature=0.3,
            max_tokens=60
        )
        authors = response.choices[0].text.strip()

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"{text[0]}\{text[0]}\n{text[0]}\n\nAcknowledgements in the paper:",
            temperature=0.3,
            max_tokens=200
        )
        acknowledgements = response.choices[0].text.strip()

        return title, authors, acknowledgements     