import speech_recognition as sr
import os, io,time, random
from pprint import pformat
from opencc import OpenCC
from itertools import chain
from datetime import datetime
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
# from pydub import AudioSegment

from transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel, BertTokenizer
import torch
import torch.nn.functional as F

from langconv import Converter # 簡繁體轉換
import soundfile
from espnet2.bin.asr_inference import Speech2Text
from pydub import AudioSegment
from pydub.silence import split_on_silence

def tokenize(obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)

cc = OpenCC('tw2s')
SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[speaker1]", "[speaker2]"]
device = torch.device("cuda")

random.seed(42)
torch.random.manual_seed(42)
torch.cuda.manual_seed(42)

tokenizer_class = BertTokenizer
# model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
model_class = OpenAIGPTLMHeadModel
tokenizer = tokenizer_class.from_pretrained("Dialog_model/0507/", do_lower_case=True)
model = model_class.from_pretrained("Dialog_model/0507/")
model.to(device)
model.eval()

## prepare translator

## prepare dialog
app = FastAPI()
step = 0
speech2text = Speech2Text('config.yaml','40epoch.pth') # ASR 模型

def silence_based_conversion(path, speech2text):
  
    song = AudioSegment.from_wav(path)
  
    # split track where silence is 0.5 seconds 
    # or more and get chunks
    chunks = split_on_silence(song,
        # must be silent for at least 0.5 seconds
        # or 500 ms. adjust this value based on user
        # requirement. if the speaker stays silent for 
        # longer, increase this value. else, decrease it.
        min_silence_len = 500,
  
        # consider it silent if quieter than -16 dBFS
        # adjust this per requirement
        silence_thresh = -20
    )
  
    # create a directory to store the audio chunks.
    try:
        os.mkdir('audio_chunks')
    except(FileExistsError):
        pass
  
    # move into the directory to
    # store the audio files.
    # os.chdir('audio_chunks')
  
    i = 0
    text = ''
    # process each chunk
    for chunk in chunks:
              
        # Create 0.5 seconds silence chunk
        chunk_silent = AudioSegment.silent(duration = 10)
  
        # add 0.5 sec silence to beginning and 
        # end of audio chunk. This is done so that
        # it doesn't seem abruptly sliced.
        audio_chunk = chunk_silent + chunk + chunk_silent
  
        # export audio chunk and save it in 
        # the current directory.
        # print("saving chunk{0}.wav".format(i))
        # specify the bitrate to be 192 k
        audio_chunk.export("./audio_chunks/chunk{0}.wav".format(i), bitrate ='192k', format ="wav")
        y, sr = soundfile.read("audio_chunks/chunk{0}.wav".format(i))
        text = text + speech2text(y)[0][0] + '，'
        # os.remove("audio_chunks/chunk{0}.wav".format(i))
        i += 1
    text = text.strip('，') + '。'
    # os.chdir('..')
    return text

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):

    assert logits.dim() == 1  
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)


        sorted_indices_to_remove = cumulative_probabilities > top_p

        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0


        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def build_input_from_segments(history, reply, tokenizer, with_eos=True):

    bos, eos, pad, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [bos] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:])
                                          for _ in s]
    return instance, sequence


def sample_sequence(history, tokenizer, model, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(70):
        instance, sequence = build_input_from_segments(history, current_output, tokenizer, with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], dtype=torch.long, device='cuda').unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], dtype=torch.long, device='cuda').unsqueeze(0)

        logits, *_ = model(input_ids, token_type_ids=token_type_ids)
        logits = logits[0, -1, :] / 0.7
        logits = top_filtering(logits, top_k=0, top_p=0.9)
        probs = F.softmax(logits, dim=-1)

        # prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        prev = torch.topk(probs, 1)[1] 
        if i < 1 and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/dialog_text/")
def read_item(user_id:str,item_id: str, q: str = None):
    history =[]
    raw_text = " ".join(list(cc.convert(item_id).replace(" ", "")))
    history.append(tokenize(raw_text))
    with torch.no_grad():
        out_ids = sample_sequence(history, tokenizer, model)
    history.append(out_ids)
    history = history[-(2 * 5 + 1):]
    out_text = Converter('zh-hant').convert(tokenizer.decode(out_ids, skip_special_tokens=True).replace(' ','')).replace('幺','麼')

    print(item_id)
    print(user_id)
    print(out_text)
    print(datetime.now())
    with open('record/dialogue_record.txt', 'a') as record_file:
        record_file.write(item_id+'_eos_'+out_text+'\n')

    return {"Sys":out_text}

@app.get("/get_wav/")
def main():
    file_path = "zhtts_wav.wav"
    return FileResponse(path=file_path, filename=file_path, media_type='text/wav')

    # client: 
    # import requests
    # r = requests.get('https://localhost:8087/get_wav/')
    # open('test.wav','wb').write(r.content)

    
@app.post("/upload_wav_file/")
async def upload(file: UploadFile = File()):
    fn = file.filename
    
    save_path = 'up/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, fn)
    f = open(save_file, 'wb')
    data = await file.read()

    f.write(data)
    f.close()

    r = sr.Recognizer()
    data_asr = sr.AudioFile(save_file)
    
    with data_asr as source:
        audio = r.record(source)
    print(audio)
    text = r.recognize_google(audio,language = 'zh-tw')

    return {'msg': f'{fn}上傳成功'}
    # client:
    # 
    # import requests
    # files1 = {'file': open('output1.wav', 'rb')}
    # r = requests.post('https://localhost:8087/upload_wav_file/', files=files1)
    # print(r.text)



@app.post("/dialog_audio/{user_id}/") 
async def upload(user_id:str, file: UploadFile = File()):

    fn = file.filename
    # 儲存路徑
    save_path = 'up/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file = os.path.join(save_path, fn)
    f = open(save_file, 'wb')
    data = await file.read()
    f.write(data)
    f.close()
    start_time = time.time()
   
    if save_file.endswith('m4a'):
        
        target_file = save_file.replace('m4a','wav')
        command = 'ffmpeg -i '+save_file+' -ac 1 -ar 16000 '+target_file
        os.system(command)
        os.remove(save_file)
        save_file = target_file
    elif save_file.endswith('mp3'):
        target_file = save_file.replace('mp3','wav')
        command = 'ffmpeg -i '+save_file+' '+target_file
        os.system(command)
        os.remove(save_file)
        save_file = target_file
    
    text = silence_based_conversion(save_file, speech2text)
    end_time = time.time()

    history =[]
    raw_text = " ".join(list(cc.convert(text).replace(" ", "")))
    history.append(tokenize(raw_text))
    with torch.no_grad():
        out_ids = sample_sequence(history, tokenizer, model)
    history.append(out_ids)
    history = history[-(2 * 5 + 1):]
    out_text = Converter('zh-hant').convert(tokenizer.decode(out_ids, skip_special_tokens=True).replace(' ','')).replace('幺','麼')

    print(user_id)
    print(text)
    print(out_text)
    print(end_time-start_time)
    print(datetime.now())
    with open('record/dialogue_record.txt', 'a') as record_file:
        record_file.write(text+'_eos_'+out_text+'_wav_'+save_file+'\n')
    # os.remove(save_file)

    return {"User":text, "Sys":out_text}
