import os
import string
import json
import torch
import numpy as np
import openai
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
#from nemo.collections.nlp.models import PunctuationCapitalizationModel
import argparse
from tqdm import tqdm
import spacy
from sentence_transformers import SentenceTransformer
import multiprocessing as mp
import _io

nlp = spacy.load('en_core_web_sm')
sent_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
f = open("/home/shang/self/openai-api.txt")
openai.api_key = f.readlines()[0]

def process_video(video_id, args, input_steps, transcripts, tokenizer, output_queue, punct_cap_model=None):
    '''Main function that processes the video. Takes in arguments:
    - video_id: id of input video
    - args: ???
    - input_steps: 
    - transcripts: transcripts, indexed by video_id
    - tokenizer:
    - punct_cap_model:
    - output_queue:
    '''
    prompt = "Write the steps of the task that the person is demonstrating, based on the noisy transcript.\nTranscript: |||1\nSteps:\n1."
    # Indexes into transcripts if argument is passed, else processes it
    #print("TRANSCRIPTS:", transcripts)
    if transcripts is not None:
        try:
            transcript = transcripts[video_id]
        except:
            return
    # Creates the output path, adds capitalization and other formatting if specified
    # Tokenizes transcript and saves it as `tokens`
    tokens = tokenizer(transcript)
    print(video_id, len(transcript), len(tokens["input_ids"]))
    while len(tokens["input_ids"]) > 1600:
        transcript = transcript[:-100]
        tokens = tokenizer(transcript)
    if args.input_steps_path is not None:
        if video_id not in input_steps:
            return
        steps = input_steps[video_id]["steps"]
    else:
        if video_id in finished:
            return
        input_text = prompt.replace("|||1", transcript)
        steps = []
        num_attempts = 0
        while len(steps) == 0:
            response = openai.Completion.create(
                            engine="text-babbage-001",
                            prompt=input_text,
                            temperature=0.7,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )
            output = response["choices"][0]["text"].strip()
            num_attempts += 1
            steps = output.split("\n")
            if all(["." in step for step in steps[1:]]):
                steps = steps[:1]+[step[step.index(".")+1:].strip() for step in steps[1:]]
            elif num_attempts < args.max_attempts:
                steps = []
    output_dict = {"video_id": video_id, "steps": steps, "transcript": transcript}
    if not args.no_align:
        #TODO: Compare similarities
        pass
    if isinstance(output_queue, _io.TextIOWrapper):
        output_queue.write(json.dumps(output_dict)+'\n')
    else:
        output_queue.put(json.dumps(output_dict)+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_list_path")
    parser.add_argument("--transcripts_path")
    parser.add_argument("--formatted_transcripts_path")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=1)
    parser.add_argument("--no_formatting", action="store_true")
    parser.add_argument("--output_path")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_align", action="store_true")
    parser.add_argument("--input_steps_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--no_dtw", action="store_true")
    parser.add_argument("--dtw_window_size", type=int, default=1000000)
    args = parser.parse_args()
    
    '''
    Specify device, CPU vs. GPU
    '''
    #print(args)

    if not args.no_align:
        if args.cpu:
            sent_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2').cpu()
        else:
            sent_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2').cuda()
    # sent_model = AutoModel.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2').cuda()

    '''
    Args no formatting - load pretrained punctuation capitalization model
    
    if not args.no_formatting:
        punct_cap_model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")
        if args.cpu:
            punct_cap_model = punct_cap_model.cpu()
    '''
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    '''
    Opens list of videos
    '''
    f = open(args.video_list_path)
    content = f.read()
    video_ids = content.split(",")
    #print(video_ids)
    '''
    Loads transcripts
    '''
    transcripts = None
    if args.transcripts_path[-5:] == ".json":
        f = open(args.transcripts_path)
        transcripts = json.load(f)
    '''
    Video End-index, can be used to truncate # videos read
    '''
    if args.end_index is not None:
        video_ids = video_ids[:args.end_index]
    '''
    Video Start-index
    '''
    video_ids = video_ids[args.start_index:]
    '''
    Ending: output is read and the video id is added to set "finished"
    '''
    finished = set()
    if os.path.exists(args.output_path):
        fout = open(args.output_path)
        written_lines = fout.readlines()
        fout.close()
        for line in written_lines:
            try:
                datum = json.loads(line)
                finished.add(datum['video_id'])
            except:
                pass
        fout = open(args.output_path, 'a')
    else:
        fout = open(args.output_path, 'w')
    '''
    Reads input_steps
    '''
    input_steps = None
    if args.input_steps_path is not None:
        f = open(args.input_steps_path)
        lines = f.readlines()
        input_steps = [json.loads(line) for line in lines]
        input_steps = {datum["video_id"]: datum for datum in input_steps}
    '''
    Goes through list of all video_ids, if video is in set finished, skip and move to next unfinished video
    '''
    for video_id in tqdm(video_ids):
        #print(video_id, finished)
        if video_id in finished:
            continue
        # job = pool.apply_async(process_video, (video_id, args, input_steps, transcripts, tokenizer, punct_cap_model, q))
        '''
        Call process_video here
        '''
        process_video(video_id, args, input_steps, transcripts, tokenizer, fout)
        # print('here', len(jobs))
        # jobs.append(job)
    fout.close()
