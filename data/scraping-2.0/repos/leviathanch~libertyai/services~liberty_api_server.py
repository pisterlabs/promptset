from typing import Any
import gc
import uuid
import sys
import time

from flask import Flask, request
from gevent.pywsgi import WSGIServer

import torch.multiprocessing as mp

import threading
import torch
from typing import Any, Dict, List, Mapping, Optional, Set

import os, copy, types, gc, sys
import numpy as np

from LibertyAI import get_configuration

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import ADA_TOKEN_COUNT

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sentence_transformers import SentenceTransformer, util

import argparse

# Threading stuff for generation job
generation_event = threading.Event()
tokens = {}
processes = {}
current_job_params = {}

# Model loading
def load_model(config):
    args = types.SimpleNamespace()
    args.RUN_DEVICE = "cuda"
    args.FLOAT_MODE = "fp16"
    os.environ["RWKV_JIT_ON"] = '1'
    os.environ["RWKV_RUN_DEVICE"] = 'cuda'
    args.RUN_DEVICE = "cuda"
    args.ctx_len=1024
    args.MODEL_NAME = "/home/user/RWKV/RWKV-4-Raven-3B-v9-Eng99%-Other1%-20230411-ctx4096"

    from rwkv.model import RWKV
    model = RWKV(
        "/home/user/RWKV/RWKV-4-Raven-3B-v9-Eng99%-Other1%-20230411-ctx4096.pth",
        'cuda:0 fp16 -> cuda:1 fp16'
        #'cuda:0 fp16i8 -> cuda:1 fp16i8'
    )
    model.share_memory()
    model.eval()

    import tokenizers
    tokenizer = tokenizers.Tokenizer.from_file("/home/user/RWKV/20B_tokenizer.json")

    from rwkv.utils import PIPELINE
    pipeline = PIPELINE(model, "/home/user/RWKV/20B_tokenizer.json")

    return model, tokenizer, pipeline

def run_rnn(model, pipeline, _tokens: List[str], newline_adj: int = 0, CHUNK_LEN: int = 256, model_tokens = [], model_state: Any = None) -> Any:
    AVOID_REPEAT_TOKENS = []
    AVOID_REPEAT = "，：？！"
    for i in AVOID_REPEAT:
        dd = pipeline.encode(i)
        assert len(dd) == 1
        AVOID_REPEAT_TOKENS += dd

    tokens = [int(x) for x in _tokens]
    model_tokens += tokens

    out: Any = None

    while len(tokens) > 0:
        out, model_state = model.forward(
            tokens[: CHUNK_LEN], model_state
        )
        tokens = tokens[CHUNK_LEN :]
    END_OF_LINE = 187
    out[END_OF_LINE] += newline_adj  # adjust \n probability

    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = -999999999

    return out, model_tokens, model_state

def has_stop_token(text, stops):
    for s in stops:
        if s in text:
            return True
    return False

def token_partial_stop(tok, stops):
    for s in stops:
        if tok in s:
            return True
    return False

last_tokens = []
def generation_job(model, tokenizer, pipeline, data):
    uid = data['uuid']
    model_tokens = []
    model_state = None
    
    tokens[uid] = []

    sem.acquire()
    logits, model_tokens, model_state = run_rnn(
        model,
        pipeline,
        tokenizer.encode(data['prompt']).ids,
        CHUNK_LEN = data['CHUNK_LEN'],
        model_tokens = model_tokens,
        model_state = model_state
    )
    sem.release()

    begin = len(model_tokens)
    out_last = begin
    decoded = ""
    occurrence: Dict = {}
    overallstring = ""

    for i in range(int(data['max_tokens_per_generation'])):
        for n in occurrence:
            logits[n] -= (
                data['penalty_alpha_presence'] + occurrence[n] * data['penalty_alpha_frequency']
            )
        sem.acquire()
        token = pipeline.sample_logits(
            logits,
            temperature=data['temperature'],
            top_p=data['top_p'],
        )
        sem.release()

        END_OF_TEXT = 0
        if token == END_OF_TEXT:
            break
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        sem.acquire()
        logits, model_tokens, model_state = run_rnn(
            model,
            pipeline,
            [token],
            CHUNK_LEN = data['CHUNK_LEN'],
            model_tokens = model_tokens,
            model_state = model_state
        )
        sem.release()
        xxx = tokenizer.decode([token])
        if token_partial_stop(xxx, data['stop']):
            last_tokens.append(xxx)
        else:
            for tok in last_tokens:
                tokens[uid].append(tok)
                sys.stdout.write(tok)
            last_tokens.clear()
            tokens[uid].append(xxx)
            sys.stdout.write(xxx)
            sys.stdout.flush()

        overallstring += xxx
        if has_stop_token(overallstring, data['stop']):
            last_tokens.clear()
            break

    tokens[uid].append("[DONE]")

def generation_worker():
    model, tokenizer, pipeline = load_model(config)
    while True:
        try:
            generation_event.wait()
            generation_job(model, tokenizer, pipeline, current_job_params)
            generation_event.clear()
        except KeyboardInterrupt:
            print("Ctrl+C pressed. Exiting...")
            break

def register_model(app):
    @app.route('/api/completion/submit', methods=['POST'])
    def completion_submit():
        if generation_event.is_set():
            return {'status': "Busy: System is busy."}

        data = request.get_json()
        if "text" not in data:
            return {'status': "Erros: No input field provided"}

        uid = str(uuid.uuid4())
        current_job_params['max_tokens_per_generation'] = int(data['max_new_tokens']) if 'max_new_tokens' in data else 256
        current_job_params['temperature'] = float(data['temperature']) if 'temperature' in data else 1.0
        current_job_params['top_p'] = float(data['top_p']) if 'top_p' in data else 0.5
        current_job_params['CHUNK_LEN'] = int(data['CHUNK_LEN']) if 'CHUNK_LEN' in data else 256
        current_job_params['penalty_alpha_frequency'] = float(data['penalty_alpha_frequency']) if 'penalty_alpha_frequency' in data else 0.4
        current_job_params['penalty_alpha_presence'] = float(data['penalty_alpha_presence']) if 'penalty_alpha_presence' in data else 0.4
        current_job_params['prompt'] = data['text']
        current_job_params['stop'] = data['stop'] if 'stop' in data else []
        current_job_params['uuid'] = uid
        tokens[uid] = []
        generation_event.set()
        return {'uuid': uid}

    @app.route('/api/completion/fetch', methods=['POST'])
    def completion_fetch():
        data = request.get_json()
        if "uuid" not in data:
            return {'text': "[DONE]"}
        uid = data["uuid"]
        if "index" not in data:
            return {'text': "[DONE]"}
        index = int(data["index"])
        while index+1 > len(tokens[uid]):
            time.sleep(1/1000)
        return {'text': tokens[uid][index]}

def embed_text(text):
    return embedding_model.encode([text])

def register_embedding(app):
    @app.route('/api/embedding', methods=['POST'])
    def embedding():
        data = request.get_json()

        try:
            text = data['text']
        except:
            return {'error': "No text provided"}

        sem.acquire()
        gc.collect()
        output = embed_text(text)
        gc.collect()
        torch.cuda.empty_cache()
        sem.release()
        return {'embedding': output[0].tolist()}

def register_sentiment(app):
    @app.route('/api/sentiment', methods=['POST'])
    def sentiment():
        data = request.get_json()
        try:
            key = data['API_KEY']
        except:
            return {'error': "Invalid API key"}

        try:
            text = data['text']
        except:
            return {'error': "No text provided"}

        if key == config.get('API', 'KEY'):
            sem.acquire()
            gc.collect()
            sent = sentiment_model.polarity_scores(text)
            gc.collect()
            torch.cuda.empty_cache()
            sem.release()
            return sent
        else:
            return {'error': "Invalid API key"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LibertyAI: API server',
        description='Choose what API services to run',
        epilog='Give me Liberty or give me death - Patrick Henry, 1775'
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-m', '--model', action='store_true')
    parser.add_argument('-e', '--embeddings', action='store_true')
    parser.add_argument('-s', '--sentiment', action='store_true')
    args = parser.parse_args()
    if args.model or args.embeddings:
        config = get_configuration()
        sem = threading.Semaphore(10)
        app = Flask(__name__)
        gc.freeze()
        gc.enable()
        if args.model:
            register_model(app)
            p = threading.Thread(target=generation_worker)
            p.start()
        if args.embeddings:
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            register_embedding(app)
        if args.sentiment:
            sentiment_model = SentimentIntensityAnalyzer()
            register_sentiment(app)
        http_server = WSGIServer(('', int(config.get('DEFAULT', 'APIServicePort'))), app)
        http_server.serve_forever()
    else:
        parser.print_help()
