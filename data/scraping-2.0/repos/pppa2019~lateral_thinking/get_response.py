from unittest import result
import openai
import os
import numpy as np
from transformers import TextGenerationPipeline
from time import sleep

from utils.metrics import bert_similarity
import time
import torch


def get_response_from_OpenAI(prompt, model=None, tokenizer=None):
    result = ""
    count = 0
    print(prompt)
    while True:
        try:
            # print('get gpt4 ][')
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                # model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            break
        except Exception as e:
            print(e)
            print("Retry in 10 seconds")
            time.sleep(10)
    # response = response["choices"][0]["message"]["content"]
    print(response.get('choices')[0]["message"]["content"])
    result_raw = response.get('choices')[0]["message"]["content"]
    result = response.get('choices')[0]["message"]["content"].strip().split("\n")[0]
    request_log = response.get('choices')[0]
    count += 1
    # if result == '':
    #     import ipdb;ipdb.set_trace()
    # sleep(5)
    # if result == "":
    #     print('Connetion failed! Sleep for 15 sec...')
    #     sleep(15)
    return result, result_raw, request_log


def get_response_from_KG(puzzle_kg_path, solution_kg_path, question, model):
    ''' 
        TODO:
        - parse question to event graph
        - fuzzy match to judge if there is event or entity overlap
    '''
    import sys
    sys.path.append('.')
    from utils.load_kg_ann import load_kg_from_ann
    from amr2brat import convert_sent2EventGraph
    def abstract_entity_event(span_dict):
        entity_list = []
        event_list = []
        for _, value in span_dict.items():
            if value[0]=='Head_End':
                if len(value)==2:
                    entity_list.append(value[1])
                else:
                    entity_list.append(' '.join(value[3:]))
            elif value[0]=='Event':
                if len(value)==2:
                    entity_list.append(value[1])
                else:
                    event_list.append(' '.join(value[3:]))
        return entity_list, event_list
    
    def abstract_event_triple(span_dict, event_dict):
        event_triple_list = []
        for value in event_dict.values():
            event_triple_list.append(' '.join([span_dict[span_id.split(':')[-1]][-1] for span_id in value]))
        return event_triple_list
    
    pz_span_dict, _, pz_event_dict = load_kg_from_ann(puzzle_kg_path)
    sl_span_dict, _, sl_event_dict = load_kg_from_ann(solution_kg_path)
    q_span_dict, _, q_event_dict = convert_sent2EventGraph(question)

    pz_entity, pz_event = abstract_entity_event(pz_span_dict)
    sl_entity, sl_event = abstract_entity_event(sl_span_dict)
    q_entity, q_event = abstract_entity_event(q_span_dict)
    pz_event_triples = abstract_event_triple(pz_span_dict, pz_event_dict)
    sl_event_triples = abstract_event_triple(sl_span_dict, sl_event_dict)
    q_event_triples = abstract_event_triple(q_span_dict, q_event_dict)

    # none event, if match entity, then yes
    threshold = 0.6
    q_entity = list(set(q_entity))
    pz_entity = list(set(pz_entity))
    story_entity = list(set(sl_event+pz_event))
    q_event_triples = list(set(q_event_triples))
    story_event_triples = list(set(sl_event_triples+pz_event_triples))
    pz_event_triples = list(set(pz_event_triples))

    simi_score = bert_similarity(q_entity, story_entity, model)
    hint_score = bert_similarity(q_entity, pz_entity, model)
    event_score = bert_similarity(q_event, story_event_triples, model)
    match_event1_id, match_event2_id = np.where(event_score>threshold)
    match_hit1_id, match_hit2_id = np.where(simi_score>threshold)
    hint_hit1_id, hint_hit2_id = np.where(hint_score>threshold)

    keyword_list = []
    for id in hint_hit1_id:
        keyword_list.append(q_entity[id])
    for id in hint_hit2_id:
        keyword_list.append(pz_entity[id])
    keyword_list = list(set(keyword_list))
    if len(pz_entity+sl_entity)==0 and len(match_hit1_id)>0:
        return 'Yes', keyword_list
    
    # have event, 

    if len(match_hit1_id)>0:
        if len(match_event1_id):
            return 'Yes', keyword_list
        return 'No', keyword_list
    
    return "Irrelevant", []


def get_response_from_chatglm(prompt, model, tokenizer):
    
    response, _ = model.chat(tokenizer, prompt, history=[])
    
    return response, None, None
        


def get_response_from_llama(prompt, model, tokenizer):
    
    pipeline = TextGenerationPipeline(model=model, batch_size=1, tokenizer=tokenizer,
                                      return_full_text=False,
                                      clean_up_tokenization_spaces=True,
                                      handle_long_generation="hole")
    
    pipeline.tokenizer.pad_token_id = model.config.eos_token_id
    with torch.no_grad():     
        hypothesis = pipeline(prompt, temperature=0.1, num_beams=4, max_length=4096, top_p=0.9, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id) 
    hypothesis = [item['generated_text'] for item in hypothesis]

    return hypothesis[0], None, None



