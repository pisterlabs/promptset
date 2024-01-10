# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from flask import Flask, request, jsonify,render_template, make_response
from flask_restful import reqparse, abort, Api, Resource
import json
from jsonschema import validate
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import re
import os
import torch
import torch.nn.functional as F
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model


application = Flask(__name__)
api = Api(application)


req_parser = reqparse.RequestParser()
req_parser.add_argument('query')
req_parser.add_argument('persona')


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

# def run():
parser = ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="dataset.json", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
parser.add_argument("--max_history", type=int, default=5, help="Number of previous utterances to keep in history")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.info(pformat(args))

if args.model_checkpoint == "":
    if args.model == 'gpt2':
        raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
    else:
        args.model_checkpoint = download_pretrained_model()


if args.seed != 0:
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


logger.info("Get pretrained model and tokenizer")
tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
model = model_class.from_pretrained(args.model_checkpoint)
model.to(args.device)
add_special_tokens_(model, tokenizer)

logger.info("Sample a personality")
dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
logger.info("Number of personality: %d", len(personalities) )
personality = personalities[2]
logger.info("Selected personality ID: %d", personalities.index(personality))
logger.info("Selected personality Desc: %s", tokenizer.decode(chain(*personality)))
history = []


class Index(Resource):    
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'),200,headers) 


class Chat(Resource):    
    def get(self):
        global history,personalities,tokenizer,model,args
        req_args = req_parser.parse_args()   
        input = req_args['query']
        logger.info("user input: %s",input)
                
        if  input =='': 
            print('input should not be empty!')
            return jsonify({'input':input,'replay':'input should not be empty!'})
        history.append(tokenizer.encode(input))
        with torch.no_grad():
            out_ids = sample_sequence(personality, history, tokenizer, model, args)
        history.append(out_ids)
        history =  history[-(2*args.max_history+1):]
        reply = tokenizer.decode(out_ids, skip_special_tokens=True)
        logger.info("reply: %s",reply)        
        return {'input':input,'reply':reply}  


class History(Resource):    
    def get(self):
        global history,tokenizer ,logger        
        hist = []
        for item in history :
            hist.append(tokenizer.decode(item))
        logger.info("History: %s", hist)
        return {'History:':hist}


class Persona(Resource):
    def get(self):
        global args
        personas={}
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
        
        for i,data in enumerate(dataset["train"]):
            personas["persona_id : "+str(i)]=data["personality"]

        return {'availabe personalities:':personas}


class Add_persona(Resource):    
    def post(self):
        try:
            persona_schema = {'type':'object',
            'properties':{
                'persona':{ 'type': 'string' },
                'back_story':{ 'type': 'string' },
                'history':{ 'type': 'string' }
                }
            }
            global logger,dataset,tokenizer, args,personalities,personality
            logger.info('input: %s',request.json)
            validate(instance = request.json,schema=persona_schema)
            persona = request.json["persona"]
            persona = re.split('\.|\?|! ',persona)
            logger.info("persona : %s",persona)

            candidates = request.json["back_story"]
            candidates = re.split('\.|\?|! ',candidates)
            logger.info("candidates : %s",candidates)

            history = request.json["history"]
            history = re.split('\.|\?|! ',history)
            logger.info("history : %s",history)
            
            new_persona={
                "personality": persona,
                "utterances": [
                    {
                        "candidates": candidates,
                        "history": history
                    }
                ]
            }

            logger.info("new persona : %s",new_persona)

            
            with open(args.dataset_path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            data["train"].append(new_persona)

            with open(args.dataset_path, "w", encoding="utf-8") as outfile:
                json.dump(data, outfile,indent=4)

            # re-initiate 
            dataset_cache = args.dataset_cache + '_' + type(tokenizer).__name__ 
            if os.path.isfile(dataset_cache):
                logger.info("removing previous cache")
                os.remove(dataset_cache)

            logger.info("Selecting your personality")
            dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
            personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]

            personality = personalities[-1]
            persona_id = personalities.index(personality)
            logger.info("Selected personality ID: %d", persona_id)
            logger.info("Selected personality Desc: %s", tokenizer.decode(chain(*personality)))

            
            return {'personality created':'success',
                    'persona_id':persona_id,
                    'persona input':persona , 
                    'candidates':candidates , 
                    'history':history,}
        except Exception as e:
            return {'error':'invalid fromat'}


class Set_persona(Resource):
    def put(self,persona_id):
        global personalities,personality,tokenizer ,logger
        persona_id = int(persona_id)
        logger.info("setting persona for ID : %d", persona_id)
        

        if persona_id > (len(personalities)-1):
            return {'persona error':'persona does not exist'}
                
        personality = personalities[persona_id]
        persona_desc= tokenizer.decode(chain(*personality))
        logger.info("Selected personality ID: %d", personalities.index(personality))
        logger.info("Selected personality: %s", persona_desc)
        return {'Selected personality':persona_desc}


class Remove_persona(Resource):
    def delete(self, persona_id):
        global logger,dataset,tokenizer, args,personalities,personality
        persona_id = int(persona_id)
        logger.info("setting persona for ID : %d", persona_id)
        
        if len(personalities)<=2:
            return {'error':'no more persona can be deleted .bot needs alteast 2 persona to work!'}
        

        if persona_id > (len(personalities)-1):
            logger.info("condition true, nop: %d", len(personalities)-1)
            return {'persona error':'persona does not exist'}
        
        
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            data = json.loads(f.read())
        del data["train"][persona_id]
        with open(args.dataset_path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile,indent=4)
        

        personas={}
        for i,data in enumerate(data["train"]):
            personas["persona_id : "+str(i)]=data["personality"]



        dataset_cache = args.dataset_cache + '_' + type(tokenizer).__name__ 
        if os.path.isfile(dataset_cache):
            logger.info("removing previous cache")
            os.remove(dataset_cache)

        logger.info("Selecting new personality")
        dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
        personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
        personality = personalities[-1]


        return {'persona_id ':persona_id,
                'status':'deleted',
                'availabe personas':personas}

class Status(Resource):
    def get(self):
        return "running"

api.add_resource(Index , '/')
api.add_resource(Chat , '/chat')
api.add_resource(History , '/history')
api.add_resource(Persona, '/persona')
api.add_resource(Add_persona,'/add_persona')
api.add_resource(Set_persona,'/set_persona/<int:persona_id>')
api.add_resource(Remove_persona,'/remove_persona/<int:persona_id>')
api.add_resource(Status,'/status')

if __name__ == "__main__":    
    application.run(debug=True)

