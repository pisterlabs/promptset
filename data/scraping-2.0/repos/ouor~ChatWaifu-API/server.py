from scipy.io.wavfile import write
from text import text_to_sequence
from models import SynthesizerTrn
import utils
import commons
import sys
import re
from pydub import AudioSegment
import torch
from torch import no_grad, LongTensor
import logging
import argparse
import requests
import json
import os
import openai
import socket
from navertts import NaverTTS
import datetime
import glob

class SocketServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self):
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.client, self.addr = self.socket.accept()

    def receive(self):
        total_data = b""
        while True:
            data = self.client.recv(1024)
            total_data += data
            if len(data) < 1024:
                break
        return total_data.decode()

    def send(self, data):
        self.client.send(data.encode())

    def stop(self):
        self.socket.close()

class vits():
    def __init__(self, model, config):
        logging.getLogger('numba').setLevel(logging.WARNING)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hps_ms = utils.get_hparams_from_file(config)
        n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
        self.n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0

        self.net_g_ms = SynthesizerTrn(
            self.n_symbols,
            hps_ms.data.filter_length // 2 + 1,
            hps_ms.train.segment_size // hps_ms.data.hop_length,
            n_speakers=n_speakers,
            **hps_ms.model).to(self.device)
        _ = self.net_g_ms.eval()
        self.hps_ms = hps_ms
        utils.load_checkpoint(model, self.net_g_ms)

    def get_text(self, text, hps, cleaned=False):
        if cleaned:
            text_norm = text_to_sequence(text, hps.symbols, [])
        else:
            text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm

    def get_label_value(self, text, label, default, warning_name='value'):
        value = re.search(rf'\[{label}=(.+?)\]', text)
        if value:
            try:
                text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
                value = float(value.group(1))
            except:
                print(f'Invalid {warning_name}!')
                sys.exit(1)
        else:
            value = default
        return value, text


    def get_label(self, text, label):
        if f'[{label}]' in text:
            return True, text.replace(f'[{label}]', '')
        else:
            return False, text

    def generateSound(self, inputString, id):
        if self.n_symbols != 0:
            text = inputString

            length_scale, text = self.get_label_value(
                text, 'LENGTH', 1, 'length scale')
            noise_scale, text = self.get_label_value(
                text, 'NOISE', 0.667, 'noise scale')
            noise_scale_w, text = self.get_label_value(
                text, 'NOISEW', 0.8, 'deviation of noise')
            cleaned, text = self.get_label(text, 'CLEANED')

            stn_tst = self.get_text(text, self.hps_ms, cleaned=cleaned)
            
            speaker_id = id
            out_path = "./output.wav"
            with no_grad():
                x_tst = stn_tst.unsqueeze(0).to(self.device)
                x_tst_lengths = LongTensor([stn_tst.size(0)]).to(self.device)
                sid = LongTensor([speaker_id]).to(self.device)
                audio = self.net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                        noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.to(self.device).cpu().float().numpy()

            write(out_path, self.hps_ms.data.sampling_rate, audio)
            print('Successfully saved!')
            # torch.cuda.empty_cache()
            return out_path

get_dir = lambda x: os.path.split(os.path.realpath(x))[0]

def download_file(url, save_dir):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(os.path.join(save_dir, local_filename), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return local_filename

class openai_session():
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        self.messages = []
        self.model = "gpt-3.5-turbo"
        self.currunt_log = f"userfile/log/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        if not os.path.exists("userfile/log"):
            os.makedirs("userfile/log")

    def save(self):
        with open(self.currunt_log, 'w', encoding='utf-8') as f:
            data = json.dumps(self.messages, ensure_ascii=False, indent=4)
            f.write(data)

    def set_role(self, role):
        prefix = "이제부터 당신은 다음과 같은 역할을 맡아 대화를 진행합니다: \n"
        self.messages.append({"role": "system", "content": prefix + role})

    def set_greeting(self, greeting):
        self.messages.append({"role": "assistant", "content": greeting})
    
    def send_message(self, message):
        try:
            self.messages.append({"role": "user", "content": message})
            res = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages if len(self.messages) <= 30 else [self.messages[0]] + self.messages[-9:],
            )
            answer = res['choices'][0]['message']['content']
            self.messages.append({"role": "assistant", "content": answer})
            self.save()
        except Exception as e:
            answer = "앗.. 뭐라고 하셨었죠? 다시 한번 말씀해 주실 수 있나요?"
            print("에러 발생: " + str(e))

        return answer
    
class navertts():
    def generateSound(self, inputString, id):
        output_path = "./output.mp3"
        tts = NaverTTS(inputString, lang='ko')
        tts.save(output_path)
        print('Successfully saved!')
        return output_path

def main():
    server = SocketServer("127.0.0.1", 9000)
    print("렌파이 클라이언트와 연결 대기중...")
    server.start()

    print("렌파이 클라이언트와 연결되었습니다.")

    tts_service = int(server.receive()) # 0: 로컬 vits, 1: 네이버

    if tts_service == 0:
        korean_model_path = r"userfile\tts\model.pth"
        korean_config_path = r"userfile\tts\config.json"

        if not os.path.isfile(korean_model_path):
            os.makedirs(get_dir(korean_model_path), exist_ok=True)
            print("TTS 모델 체크포인트 파일이 없습니다.해당 파일을 다운로드 받습니다.")
            url = 'https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/6/model.pth'
            download_file(url, get_dir(korean_model_path))
            print("TTS 모델 체크포인트 파일 다운로드 완료")

        if not os.path.isfile(korean_config_path):
            os.makedirs(get_dir(korean_config_path), exist_ok=True)
            print("TTS 모델 설정 파일이 없습니다.해당 파일을 다운로드 받습니다.")
            url = 'https://huggingface.co/spaces/skytnt/moe-tts/resolve/main/saved_model/6/config.json'
            download_file(url, get_dir(korean_config_path))
            print("TTS 모델 설정 파일 다운로드 완료")
        
        tts = vits(korean_model_path, korean_config_path)
        config = json.load(open(korean_config_path, 'r'))
        spk_list = config['speakers']
        speaker = int(server.receive())
        print("선택된 음성: " + spk_list[speaker])
    
    elif tts_service == 1:
        tts = navertts()
        speaker = 0

    print("렌파이에서 API KEY를 입력해주세요.")
    print("API KEY는 https://platform.openai.com/account/api-keys 에서 발급할 수 있습니다.")

    session_token = server.receive()

    if(session_token):
        print(f"API KEY: ...{session_token[-8:]}")
        oai = openai_session(session_token)

        setting = server.receive()
        oai.set_role(setting)
        print("배경 설정: "+ setting)

        greeting = server.receive()
        oai.set_greeting(greeting)
        print("인사말: "+ greeting)

    while True:
        question = server.receive()
        print("Question Received: " + question)

        answer = oai.send_message(question)
        print("ChatGPT:", answer)

        tts_audio_path = tts.generateSound(answer, speaker)

        # convert wav to ogg
        src = tts_audio_path
        dst = "./ChatWithGPT/game/audio/test.ogg"
        sound = getattr(AudioSegment, f'from_{src.split(".")[-1]}')(src)
        sound.export(dst, format="ogg")

        # send response to UI
        server.send(answer)

        # finish playing audio
        print(server.receive())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("종료합니다.")
        sys.exit(0)
    except ConnectionResetError:
        print("클라이언트와의 연결이 끊겼습니다.")
        sys.exit(0)