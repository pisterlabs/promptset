import openai
import json 
import re
import csv
from openai import OpenAI
client = OpenAI()

'''
Code to convert VQA why-question examples from the why-question to the declarative form, extract the lemma, and 
prepare the data for UDS annotation HIT. 
'''

# CSV outpath
out_path = 'test'
formated_data = []
hit_file_format_version = 1

# Process VQA why-questions
data = []
f = open('../100_vqa_examples.json', 'r')
data = json.load(f)
to_write = []

for i, question_id in enumerate(data): 

    url_base = "https://ugrad.cs.jhu.edu/~jgualla1/"

    append_num = 12 - len(str(question_id)[:-3])
    zero_append = append_num * '0'
    print(question_id)
    cur_data = data[question_id]
    print(cur_data)
    image_url = f"{url_base}{'COCO_train2014_'}{zero_append}{question_id[:-3]}{'.jpg'}" 
    print(image_url)   

   
    cur_why_question = cur_data["question"]
    print(f"Why question: {cur_why_question}")
    #The declarative form of \"Why are there so many birds in the sky\" is \"There are so many birds in the sky\". The declarative form of \"Why are there so many apples in the store\" is \"There are so many apples in the store\". \n The declarative form of \"" + cur_why_question + "\" is

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "The declarative form of \"Why are there so many birds in the sky\" is \"There are so many birds in the sky\". Give only the declarative forms of the question and nothing else: " + cur_why_question}
        ]
    )
    #print(completion['choices'][0]['message']['content'])


    #print(response_re)
    #choices = response_re["choices"]
    #text = choices[-1]["text"]
    #print(f"Generated text: {text}")
    #subjects = re.findall(r'"(.*?)"', text)
    declarative = response.choices[0].message.content
    print(f"Declarative: {response.choices[0].message.content}")
    declarative_tokens = declarative.split(' ').copy()


    arg_prompt = "The subject of the sentence \"The man walking\" is \"The man\". The subject of the sentence \"The birds are here\" is \"The birds\". The subject of the sentence \"The police are running\" is \"The police\". The subject of the sentence \"The child is sad\" is \"The child\". The subject of the sentence \"The are only birds in the sky\" is \"birds in the sky\". \n Give just the subject of the sentence and nothing else: \""

    #cur_why_question = cur_data["question"]
    #print(f"Why question: {cur_why_question}")

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": arg_prompt + declarative}
        ]
    )
    #print(completion.choices[0].message['content'])


    #print(f"Prompt for subject extraction: {arg_prompt + cur_why_question}")
    #choices = response_arg["choices"]
    #text = choices[-1]["text"]
    #print(f"Generated text: {text}")
    #subjects = re.findall(r'"(.*?)"', text)
    subject = response.choices[0].message.content
    subject = subject.replace("\"", "")
    subject = subject.replace(".", "")
    print(f"Subject: {subject}")


    pred_prompt = "The predicate of the sentence \"The tree is blue\" is \"is blue\". The predicate of the sentence \"The man is cooking so many hot dogs\" is \"is cooking\". The predicate of the sentence of \"The clothes are on the chair\" is \"are on the chair\". The predicate of the sentence of \"The men are walking\" is \"are walking\". The predicate of the sentence of \"The men appear only as they are\" is \"the men appear only as\".\n Give only the predicate of the sentence: \""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": pred_prompt + declarative}
        ]
    )
    #print(completion.choices[0].message)

    #print(f"Prompt for predicate extraction: {pred_prompt + cur_why_question}")
    #choices = response_pred["choices"]
    #text = choices[-1]["text"]
    ##print(f"Generated text: {text}")
    #predicates = re.findall(r'"(.*?)"', text)
    #predicate = predicates[0]
    predicate = response.choices[0].message.content
    predicate = predicate.replace("\"", "")
    predicate = predicate.replace(".", "")
    print(f"Predicate: {predicate}")

    # Locate predicate tokens
    predicate_split = predicate.split(' ').copy()
    print(declarative_tokens)
    print(predicate_split)
    predicate_loc_s = declarative_tokens.index(predicate_split[0])
    predicate_loc_e = len(predicate_split) - 1
    try:
        predicate_loc_e = declarative_tokens.index(predicate_split[-1] + '.')
    except: 
        try:
            predicate_loc_e = declarative_tokens.index(predicate_split[-1])
        except:
            predicate_loc_e = declarative_tokens.index(predicate_split[-1] + '\n')
    predicate_loc = (predicate_loc_s, predicate_loc_e)

    lemma_prompt = "The lemma of \"is walking\" is \"to walk\". The lemma of \"is brown\" is \"to be brown\". Give only the lemma of: \""
    
    # Extract lemma
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": lemma_prompt + predicate}
        ]
    )


    #print(f"Generated text: {response_lemma}")
    #choices = response_lemma["choices"]
    #text = choices[-1]["text"]
    #lemmas = re.findall(r'"(.*?)"', text)
    #lemma = lemmas[0]
    lemma = response.choices[0].message.content
    lemma = lemma.replace("\"", "")
    lemma = lemma.replace(".", "")
    print(f"Lemma: {response.choices[0].message.content}")

    pp_prompt = "The present progressive of \"to walk\" is \"walking\". The present progressive of \"to be brown\" is \"being brown\". The present progressive of \"to be in the car\" is \"being in the car\". \n Give only the present progressive of the sentence: \""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[ 
            {"role": "system", "content": pp_prompt + lemma}
        ]
    )

    #print(f"Generated text: {response_lemma}")
    #choices = response_pp["choices"]
    #text = choices[-1]["text"]
    #pp = re.findall(r'"(.*?)"', text)
    #lemma = lemmas[0]
    pp = response.choices[0].message.content
    pp = pp.replace("\"", "")
    pp = pp.replace(".", "")
    print(f"Present progressive: {pp}")

    print("------------------------------------------------")

    dic = {"argument_phrase":subject,"full_argument_label":"nsubj","sentence":'<span class=\\\\\\"argument\\\\\\"class=\\\\\\"nsubj\\\\\\">'+subject+'</span><span class=\\\\\\"predicate\\\\\\">' + predicate + '</span>'}
    
    print('<span class=\\\\\\\'argument\\\\\\\'class=\\\\\\\'nsubj\\\\\\>')

    

    line_dict = {
        'hit_file_format_version': '2.0.0', 
        'corpus_id': 'VQA', # TBD 
        'video_id': image_url, # Sentence ID
        'predicate_token_id': str(predicate_loc), # Position of predicate in sentence
        'roleset': '', # Nothing
        'predicate_lemma': lemma, # Lemma
        'predicate_progressive': pp, # Progressive
        'argnum': subject, # TBD
        'sentences_and_args_as_json': dic,
        'sampling_method': 'it-happened' # TBD
    }
    #print(line_dict)
    to_write.append(line_dict)
    #print(to_write)
    if i == 20:
        break



# Format for csv
# hit_file_format_version, corpus_id, sentence_id, predicate_token_id, roleset, predicate_lemma, predicate_progressive, argnum, sentences_and_args_as_json, sampling_method

with open("../20_input_uds_with_images.csv", "w") as f1:
    fieldnames = ['hit_file_format_version', 'corpus_id', 'video_id', 'predicate_token_id', 'roleset', 'predicate_lemma', 'predicate_progressive', 'argnum', 'sentences_and_args_as_json', 'sampling_method']
    writer = csv.DictWriter(f1, fieldnames=fieldnames)
    writer.writeheader()
    #writer.writeheader()
    for line in to_write:
        writer.writerow(line)






