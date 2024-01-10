from pprint import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter
import openai
import os
from os import path
import json
import re 
from langchain.embeddings.openai import OpenAIEmbeddings
from pod_utils import get_podcasts


def clean_transcript(file_path, guest):
    with open(file_path, 'r') as f:
        transcript = json.load(f)


    def set_speaker(transcript, guest):
        pete_speaker = transcript[0]['speaker']
        guest_speaker = transcript[1]['speaker'] if len(transcript) > 1 else None

        for segment in transcript:
            if segment['speaker'] == pete_speaker:
                segment['speaker'] = 'Peter Attia'
            elif segment['speaker'] == guest_speaker:
                segment['speaker'] = guest


    def correct_name(text):
        text = text.replace('Atiya', 'Attia')
        text = text.replace('Atiyah', 'Attia')
        text = text.replace('Attiya', 'Attia')
        text = text.replace('Attiyah', 'Attia')
        text = text.replace('Attiah', 'Attia')
        text = text.replace('Atiah', 'Attia')
        return text


    def remove_double_spaces(transcript):
        for segment in transcript:
            segment['text'] = segment['text'].replace('  ', ' ')
            segment['text'] = correct_name(segment['text'])
            if segment['text'][0] == " ":
                segment['text'] = segment['text'][1:]

            segment['words'] = [(correct_name(word), start)
                                for word, start in segment['words']]


    set_speaker(transcript, guest)
    remove_double_spaces(transcript)


    with open(file_path, 'w') as f:
        json.dump(transcript, f)


    text = ""
    for line in transcript:
        text += line['speaker'] + ": " + line["text"] + "\n"

    output_filename = path.join(
            path.dirname(file_path), "transcript.txt")
    with open(output_filename, "w") as file:
        file.write(text)

    return transcript, text    


def chunk_docs(podcast, file_path, guest, force=False):

    chunk_file_name = path.join(path.dirname(file_path), "chunks.json")
    if not force and path.exists(chunk_file_name):
        print("chunk.json exists, loading from file")
        with open(chunk_file_name, "r") as f:
            return json.load(f)

    transcript, transcript_text = clean_transcript(file_path, guest)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(transcript_text)
    


    def get_start_of_chunk_index(s,l):
        len_s = len(s) #so we don't recompute length of s on every iteration
        for i in range(len(l) - len_s+1):
            if s == l[i:len_s+i]:
                return i
        """ return any(s == l[i:len_s+i] for i in range(len(l) - len_s+1)) """

    words = []
    beginnings = []

    split_characters = r'[\s\n]|\\n'

    for t in transcript:
        speaker_chunks = re.split(split_characters, (t["speaker"]+":"))
        words.extend(speaker_chunks)
        [beginnings.append(t["words"][0][1]) for i in range(len(speaker_chunks))]
        for word, begin in t["words"]:
            words.append((word).strip())
            beginnings.append(begin)

    chunks = []

    for text in texts:
        
        chunk_words = re.split(split_characters, text)

        start_index = get_start_of_chunk_index(chunk_words, words)
        if start_index == None:
            throw("chunk not found in transcript")
            continue
        start = beginnings[start_index]
        #print("start", start, chunk_words)
        chunks.append({"meta":{"start": start, **podcast}, "text": text})
    with open(chunk_file_name, "w") as f:
        json.dump(chunks, f)
        
    return chunks

podcasts = get_podcasts()

for podcast in podcasts:
    file_path = os.path.join("data/thedrive/"+podcast["folder"]+"/transcript.json")
    if not path.exists(file_path):
        print("transcript.json does not exist for", podcast["title"])
        continue
    print("processing", podcast["title"])
    chunks = chunk_docs(podcast, file_path, podcast["guest"], True)