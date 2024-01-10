import pandas as pd
import json
import os 
import io
import openai
from tqdm import tqdm
from collections import Counter

import spacy
nlp=spacy.load('en_core_web_sm')

from retry import retry



SEGMENT_DATASET="../data/preprocessed/movie_synopsis_segments_n2.jsonl"
OUTPUT_DATASET="../data/preprocessed/movie_synopsis_segments_n2_after.jsonl"
FINAL_DATASET='../data/moral_dataset/inference_n2.tsv'


def load_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(input_path, datasets):
    with open(input_path, 'w') as outfile:
        json.dump(datasets, outfile)

# transformation API
@retry(Exception, tries=5, delay=1, backoff=2, max_delay=120)
def chatgptANW(input_text):
    user_input_script=input_text

    openai.api_key="YOUR_API_KEY"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature = 1,
        max_tokens=512,
        messages=[
            {"role": "system", "content": "Please extract one main action of the protagonist in the provided segment. \
                After that, extract Situation, Intention, and Consequence related to the action. \
                The candidate protagonist is within the brackets []. \
                If any of the categories are not mentioned in the segment, please output '0' for that category. \
                Situation: Explain the context or situation in which the action takes place. \
                Intention: Elaborate on the protagonist's intention or motivation behind this action. \
                Consequence: Describe the outcome or consequence of the action. \
                Your system have to strictly adhere to the following output format."},
            {"role": "user", "content": "Input: [char] goes to the funeral for his friend [Ronnie], and in the process of eulogizing him, while commenting on the shortness of life and love, he spontaneously proposes to [Lisa]. [char] conference calls [Sam] and [Archie] to tell them he is getting married that weekend in Las Vegas, and they not only want to come to the wedding, they want to throw him a bachelor party the night before."},
            {"role": "assistant", "content": "|Action| [char] proposes to Lisa during Ronnie's funeral. <sep> |Situation| The funeral for [char]'s friend Ronnie is taking place. [char] takes the opportunity during the eulogy to propose to Lisa.<sep> |Intention| [char]'s intention is to express his love for Lisa and commit to a lifelong relationship with her. His proposal is driven by the emotions and reflections stirred up by Ronnie's funeral.<sep> |Consequence| 0 <sep>"},
            {"role": "user", "content": "Input: [char] and [Miller] fight hand to hand. [Miller] is killed when [Ryan] impales him backward on a boat anchor, and his body is obliterated in the subsequent explosion of the craft."},
            {"role": "assistant", "content": "|Action| [char] impales [Miller] backward on a boat anchor, causing his death. <sep> |Situation| [char] and [Miller] are engaged in a hand-to-hand fight. The fight takes place on a boat. <sep> |Intention| 0 <sep> |Consequence| [Miller] is killed as a result of being impaled on the boat anchor, and his body is obliterated when the boat explodes afterward.<sep>"},
            {"role": "user", "content": "Input: "+ user_input_script}
        ]
    )

    output_text = response["choices"][0]["message"]["content"]
    return output_text


def extract_character_from_subject(sentence, main_characters):
    sentence=sentence.replace("[", "")
    sentence=sentence.replace("]", "")
    
    doc = nlp(sentence)
    
    main_char_list=[]
    for token in doc:
        #print(token.dep_, ":",token.text)
        if 'subj' in token.dep_:
            main_char_list.append(token.text)
    
    if len(main_char_list)==0:
        return '0'
    else:
        for main in main_char_list:
            if main in main_characters:
                return main
        return '0'
    
    

##############Transform##############

datasets=load_jsonl(SEGMENT_DATASET)[0]
#outputs=load_jsonl(OUTPUT_DATASET)[0]

# Transformation
outputs=[]
for i in tqdm(range(0,len(datasets))):
    segments=datasets[i]["segments"]
    seg_char=datasets[i]['segment_char']
    main_char=datasets[i]['main_char']
    movie_id=datasets[i]['movie_id']
    movie_genre=datasets[i]['genre']

    output_segments=[]
    output_prompts=[]

    for i in range(len(segments)):
        output_seg=chatgptANW(segments[i])

        if "<sep>" in output_seg:
            prompts=[x.strip() for x in output_seg.split("<sep>")]
        else:
            prompts=[x.strip() for x in output_seg.split("\n")]
        
        output_dict={}
        
        for prompt in prompts:
            if "|" not in prompt:
                continue
            else:
                try:
                    key, value=prompt.split("|")[1:]
                    if key.strip() in output_dict.keys():
                        output_dict[key.strip()]+=" | "+value.strip()
                    else:
                        output_dict[key.strip()]=value.strip()
                except ValueError:
                    print("Value Error")
                    continue
        
        for k in ["Action", "Situation", "Intention", "Consequence"]:
            if k in output_dict.keys():
                if len(output_dict[k])<10 or "Incomplete" in output_dict[k] or 'None' in output_dict[k]:
                    output_dict[k]="0"
            else:
                output_dict[k]="0"

        if len(output_dict)==4:
            output_prompts.append(output_dict)
            output_segments.append(segments[i])

    assert len(output_segments)==len(output_prompts)

    outputs.append({
        "movie_id": movie_id,
        "genre": movie_genre,
        "segments": output_segments,
        "convert_segments": output_prompts,
        "seg_char":seg_char,
        "main_char": main_char
    })

# Transformation Segment Conversion
for i in range(len(outputs)):
    #print(i)
    convert_segments=outputs[i]['convert_segments']
    main_characters=[x[0] for x in outputs[i]['main_char']]
    seg_main_character_list=[]

    for j in range(len(convert_segments)):
        conv_segments=convert_segments[j]['Action'].split(' | ')

        # action length==1
        if len(conv_segments)==1:
            if conv_segments[0]=='0': # no action
                seg_main_character_list.append('0')
                print("no action sentence")
            else:
                # action
                main_char=extract_character_from_subject(conv_segments[0], main_characters)
                print(main_char)
                if main_char=='0':
                    seg_main_character_list.append(main_char)
                else:
                    seg_main_character_list.append(main_char)
                    convert_segments[j]['Action']=convert_segments[j]['Action'].replace(main_char, 'char')
                    convert_segments[j]['Situation']=convert_segments[j]['Situation'].replace(main_char, 'char')
                    convert_segments[j]['Intention']=convert_segments[j]['Intention'].replace(main_char, 'char')
                    convert_segments[j]['Consequence']=convert_segments[j]['Consequence'].replace(main_char, 'char')

        else:
            main_character=[]
            for k in range(len(conv_segments)):
                if conv_segments[k]=='0':
                    main_character.append('0')

                else:
                    main_char=extract_character_from_subject(conv_segments[k], main_characters)
                    main_character.append(main_char)
                        

            main_main_char=Counter(main_character).most_common(1)[0][0]
            seg_main_character_list.append(main_main_char)
            convert_segments[j]['Action']=convert_segments[j]['Action'].replace(main_main_char, 'char')
            convert_segments[j]['Situation']=convert_segments[j]['Situation'].replace(main_main_char, 'char')
            convert_segments[j]['Intention']=convert_segments[j]['Intention'].replace(main_main_char, 'char')
            convert_segments[j]['Consequence']=convert_segments[j]['Consequence'].replace(main_main_char, 'char')

    assert len(seg_main_character_list)==len(convert_segments)
    
    outputs[i]['segment_main_char']=seg_main_character_list


# Final output making
final_output=[]
id=0
for i in range(len(outputs)):
    movie_id=outputs[i]['movie_id']
    genre=outputs[i]['genre']
    segments=outputs[i]['segments']
    convert_segments=outputs[i]['convert_segments']
    main_char=outputs[i]['main_char']
    segment_main_char=outputs[i]['segment_main_char']

    for seg, conv_seg, seg_char in zip(segments, convert_segments, segment_main_char):
        final_output.append({
            'ID': id,
            'norm': 'norm',
            'situation': conv_seg['Situation'],
            'intention': conv_seg['Intention'],
            'action': conv_seg['Action'],
            'consequence': conv_seg['Consequence'],
            'label': '0',
            'movid_id': movie_id,
            'original': seg,
            'genre': genre,
            'main_char': seg_char
        })
        id+=1

df = pd.DataFrame(final_output)
df.to_csv(FINAL_DATASET, sep='\t', index=False)
