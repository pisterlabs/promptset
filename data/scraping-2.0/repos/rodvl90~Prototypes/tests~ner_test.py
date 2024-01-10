import re
import uuid
from pprint import pprint

from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)


def get_entities(text):
        print("Text: ","-"*50)
        pprint(text)
        entities = meta_pipeline(text)
        print("Entities: ","-"*50)
        pprint(entities)

        entities = process_bert_base_entities(entities)
        print("Processed Entities: ","-"*50)
        pprint(entities)
        return entities
def clear_txt( text):
        # This pattern will match any character that is not a letter, a number or a space
        pattern = r'[^a-zA-Z0-9\s]'
        # re.sub() function replaces the matches with an empty string
        text = re.sub(pattern, '', text)
        return text
def get_vector_data(text):
        batch_size = 10
        chunks = to_chunks(text)
        for x in chunks:
            entities = get_entities(clear_txt(x['text']))
        
        for i in tqdm(range(0, len(chunks), batch_size)):
            # find end of batch
            i_end = min(len(chunks), i+batch_size)

            meta_batch = chunks[i:i_end]
            # get texts to encode
            texts = [x['text'] for x in meta_batch]
            
            meta_batch = [{
                'text': x['text'],
                'entities': get_entities(x['text'])
            } for x in meta_batch]


            

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
def to_chunks(text):
        chunks = []
        texts = text_splitter.split_text(text)
        chunks.extend([{
            'id': str(uuid.uuid4()),
            'text': texts[i].replace("\n", " ").replace("\\n", " ").replace("'", " "),
            'chunk': i,
        } for i in range(len(texts))])
        return chunks
def meta_pipeline(text):
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    pipe = pipeline("ner", model=model, tokenizer=tokenizer)
    return pipe(text)

def process_bert_base_entities(entities):
    current_entity = previous_entity = None
    processed_entities = {}
    word = word_entity = ""
    
    for entity in entities:
        same_entity = False
        current_entity = entity
        
        # For the first iteration, initialize the word and entity type
        if previous_entity is None:
            previous_entity = entity
            word = entity["word"].replace('##', '')
            word_entity = entity["entity"][2:]
            continue
        if (current_entity["start"] == previous_entity["end"] + 1) or (current_entity["start"] == previous_entity["end"]):
            same_entity = True
        # If current token is continuation of previous token or if it's the same entity type
        if (same_entity) or (current_entity["entity"].startswith('I-') and previous_entity["entity"][2:] == current_entity["entity"][2:]):
            word += entity["word"].replace('##', '')
        else:
            # When we encounter a new word, append the previous word to the list under its entity type
            if word_entity not in processed_entities:
                processed_entities[word_entity] = []
            processed_entities[word_entity].append(word)
            # Initialize for the new word
            word = entity["word"].replace('##', '')
            word_entity = entity["entity"][2:]

        previous_entity = entity
    
    # Add the last word after the loop
    if word_entity not in processed_entities:
        processed_entities[word_entity] = []
    processed_entities[word_entity].append(word)
    # remove duplicates from each entity list
    for entity in processed_entities:
        processed_entities[entity] = list(set(processed_entities[entity]))
    
    return processed_entities
                

          

# open getting_started.md file and get vector data
with open("getting_started.md", "r", encoding="utf-8") as f:
     vector = get_vector_data(f.read())