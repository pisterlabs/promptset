import torch
import openai
import whisper
import queue
import speech_recognition as sr
import threading
import numpy as np
import time

class RealTime_STT:
    def __init__(self,model="base",device=("cuda" if torch.cuda.is_available() else "cpu"),\
                 energy=300,pause=0.5,save_file=False,model_root="./.cashe/whisper",mic_index=None):
        self.energy = energy
        self.pause = pause
        self.save_file = save_file # 아직 안쓰임
        self.options = {'task': 'transcribe', 'language': 'Korean'}

        self.audio_model = whisper.load_model(model, download_root=model_root).to(device)
        self.audio_queue = queue.Queue()
        self.result_queue: "queue.Queue[str]" = queue.Queue()

        self.break_threads = False
        self.banned_results = [""," ","\n",None]

        self.setup_mic(mic_index)


    def setup_mic(self, mic_index):
        if mic_index is None:
            print("No mic index provided, using default")
        self.source = sr.Microphone(sample_rate=16000, device_index=mic_index)

        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy
        self.recorder.pause_threshold = self.pause

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=3)
        print("Mic setup complete, Listening for wakeup word")

    def preprocess(self, data):
        return torch.from_numpy(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0)

    def get_all_audio(self, min_time: float = 2.):
        audio = bytes()
        got_audio = False
        time_start = time.time()
        while not got_audio or time.time() - time_start < min_time:
            while not self.audio_queue.empty():
                audio += self.audio_queue.get()
                got_audio = True

        data = sr.AudioData(audio,16000,2)
        data = data.get_raw_data()
        return data

    def record_callback(self,_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        self.audio_queue.put_nowait(data)

    def transcribe_forever(self) -> None:
        while True:
            if self.break_threads:
                break
            self.transcribe()

    def transcribe(self,data=None) -> None:
        if data is None:
            audio_data = self.get_all_audio()
        else:
            audio_data = data
        audio_data = self.preprocess(audio_data)

        result = self.audio_model.transcribe(audio_data, **self.options)
        predicted_text = result["text"]
        if predicted_text not in self.banned_results:
            self.result_queue.put_nowait(predicted_text)

    def listen_loop(self) -> None:
        threading.Thread(target=self.transcribe_forever).start()
        while True:
            result = self.result_queue.get()
            print(result)

    def listen(self, timeout: int = 4):
        audio_data = self.get_all_audio(timeout)
        self.transcribe(data=audio_data)
        while True:
            if not self.result_queue.empty():
                return self.result_queue.get()



class Whisper_STT(RealTime_STT):
    def __init__(self, model="base", device=("cuda" if torch.cuda.is_available() else "cpu"),\
                 energy=300, pause=0.5, save_file=False, model_root="./.cache/whisper", mic_index=None,\
                 wakeup_word='Hello 이루멍'):
        super().__init__(model, device, energy, pause, save_file, model_root, mic_index)
        self.wakeup_word = wakeup_word
        self.destination_list = ['미래관, 정문, 후문, 본관, 학생회관, 시대융합관, 창공관']
        self.WAKEUP_WORD_PROMPT = f"너는 {self.wakeup_word}이라는 wakeup-word만 탐지해. \
                                   입력값 중에 {self.wakeup_word}과 같거나 유사한 게 있으면 Yes를 대답. \
                                   이외의 입력값은 No로 대답"
        self.PROMPT_FOR_SYSTEM = f"너는 한국말로 듣고 말하는 서울시립대학교 캠퍼스 홍보 및 길 안내 로봇이다.\
                                  If 가고 싶은 곳이 {self.destination_list}에 있으면 목적지만 알아내서 'M:목적지' 형태로 대답. \
                                  Else if 빨리 가라는 내용이면 'SF'라고만 대답.\
                                  Else if 천천히 가라는 내용이면 'SL'라고만 대답.\
                                  Else if 입력값은 'Q:입력값' 형태로 똑같이 대답."
        
    def chat_completion(self, model, temp, messages):
        return openai.ChatCompletion.create(
            model=model,
            temperature=temp,
            messages=messages
    )['choices'][0]['message']['content']

    # STT_Agent의 detect_wakeup_word 메소드
    def detect_wakeup_word(self, predicted_text):
        messages = [
            {"role": "system", "content": self.WAKEUP_WORD_PROMPT},
            {"role": "user", "content": predicted_text},
        ]
        return self.chat_completion("gpt-3.5-turbo", 0, messages) == "Yes"

    # STT_Agent의 preprocess_transcript 메소드
    def Question_transcript(self, predicted_text):
        messages = [
            {"role": "system", "content": self.PROMPT_FOR_SYSTEM},
            {"role": "user", "content": "미래관으로 가줘"},
            {"role": "assistant", "content": "M:미래관"},
            {"role": "user", "content": "속도좀 줄여줘"},
            {"role": "assistant", "content": "SL"},
            {"role": "user", "content": "동아리는 뭐가 있어?"},
            {"role": "assistant", "content": "Q:동아리는 뭐가 있어?"},
            {"role": "user", "content": predicted_text},
        ]
        return self.chat_completion("gpt-3.5-turbo", 0, messages)

    def listen_and_detect(self):
        while True:  # 끊임없이 마이크에서 음성을 인식
            print('---------------------------------')
            audio_data = self.get_all_audio()
            audio_data = self.preprocess(audio_data)
            result = self.audio_model.transcribe(audio_data, initial_prompt=self.wakeup_word)
            predicted_text = result["text"]

            if predicted_text not in self.banned_results:
                if self.detect_wakeup_word(predicted_text):
                    print("Wakeup word detected!: ", predicted_text)
                    # 사용자가 호출 명령어를 말한 뒤에 전달하고 싶은 말을 하는 시간을 주기 위해 잠시 대기
                    time.sleep(0.2)
                    
                    # 이후 전달하고 싶은 말을 듣고 처리
                    print("Listening...")
                    audio_data = self.get_all_audio()
                    audio_data = self.preprocess(audio_data)
                    result = self.audio_model.transcribe(audio_data)
                    predicted_Q = result["text"]
                    
                    if predicted_Q not in self.banned_results:
                        processed_Q = self.Question_transcript(predicted_Q)
                        print("Input Message: ", processed_Q)
                        return processed_Q
                    break  # 호출 명령어가 감지되면 루프를 종료
                else:
                    print("No wakeup word detected. Input :", predicted_text)
                    
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    agent = Whisper_STT()
    text = agent.listen_and_detect()
    print(text)
    