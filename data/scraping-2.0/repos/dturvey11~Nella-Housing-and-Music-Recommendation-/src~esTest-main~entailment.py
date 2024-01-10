#ENTAILMENT MODEL 
# get llm response from chat gpt for all 8 attributes 

import json
import openai

def get_llm_response(llm_prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.2,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            timeout=5
            # stop="stop"
        )
    except Exception as e:
        print("Exception {}".format(e))
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.2,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0.2,
            presence_penalty=0.2,
            timeout=5
            # stop="stop"
        )
    return response["choices"][0]["message"]["content"]


def entailment_llm_prompt(desc, usr_utt): 
    llm_prompt = "Given 2 utterances , 'UTTERANCE_1 and UTTERANCE_2 delimited by triple backticks, compute entailment score between them. Entailment score is a scale between 0 and 1. Entailment score is closer to 1 when UTTERANCE_1 entails UTTERANCE_2. Entailment score is closer to 0 when UTTERANCE_1 contradicts UTTERANCE_2. Entailment score is closer to 0.5 when UTTERANCE_1 does not contradict or entail UTTERANCE_2. Response should be a json string in the following format delimited by single backticks : `{\"score\": NUMBER}` Example : UTTERANCE_1 : ```I'm in the mood for some music performed live in front of an audience.```UTTERANCE_2 : ```I like heavy metal concerts``` Output : {\"score\": 0.95}  Reasoning : \"Utterance 2 talks about liking concerts, which is entailed by Utterance 1\"  UTTERANCE_1 : ```" + desc + "``` UTTERANCE_2 : ```" + usr_utt + "```" 
    return llm_prompt

def get_dancibility_entailment(utt):
    dance = "I want a song that I can dance to."

    llm_prompt = entailment_llm_prompt(dance, utt)

    resp_json = get_llm_response(llm_prompt)
    resp_dict = json.loads(resp_json)

    if(resp_dict["score"] != 0.5):
        return {
            "attribute": "dancibility",
            "match": False,
            "boolv": False,
            "phrase": None,
            "range": True,
            "val": resp_dict["score"]
        }
    else: 
        return None 

def get_energy_entailment(utt):
    energy = "I want a high energy song."

    llm_prompt = entailment_llm_prompt(energy, utt)

    resp_json = get_llm_response(llm_prompt)
    resp_dict = json.loads(resp_json)

    if(resp_dict["score"] != 0.5):
        return {
            "attribute": "energy",
            "match": False,
            "boolv": False,
            "phrase": None,
            "range": True,
            "val": resp_dict["score"]
        }
    else: 
        return None 


def get_loudness_entailment(utt):
    loudness = "I want a loud song."

    llm_prompt = entailment_llm_prompt(loudness, utt)

    resp_json = get_llm_response(llm_prompt)
    resp_dict = json.loads(resp_json)

    if(resp_dict["score"] != 0.5):
        return {
            "attribute": "loudness",
            "match": False,
            "boolv": False,
            "phrase": None,
            "range": True,
            "val": resp_dict["score"]
        }
    else: 
        return None 

def get_acousticness_entailment(utt):
    acoustic = "I'm in the mood for some music with an unplugged, acoustic sound."

    llm_prompt = entailment_llm_prompt(acoustic, utt)

    resp_json = get_llm_response(llm_prompt)
    resp_dict = json.loads(resp_json)

    if(resp_dict["score"] != 0.5):
        return {
            "attribute": "acousticness",
            "match": False,
            "boolv": False,
            "phrase": None,
            "range": True,
            "val": resp_dict["score"]
        }
    else: 
        return None 


def get_instrumentalness_entailment(utt):
    inst = "I'm looking for instrumental music to listen to."

    llm_prompt = entailment_llm_prompt(inst, utt)

    resp_json = get_llm_response(llm_prompt)
    resp_dict = json.loads(resp_json)

    if(resp_dict["score"] != 0.5):
        return {
            "attribute": "instrumentalness",
            "match": False,
            "boolv": False,
            "phrase": None,
            "range": True,
            "val": resp_dict["score"]
        }
    else: 
        return None 
    
def get_liveness_entailment(utt):
    live = "I'm in the mood for some music performed live in front of an audience."

    llm_prompt = entailment_llm_prompt(live, utt)

    resp_json = get_llm_response(llm_prompt)
    resp_dict = json.loads(resp_json)

    if(resp_dict["score"] != 0.5):
        return {
            "attribute": "liveness",
            "match": False,
            "boolv": False,
            "phrase": None,
            "range": True,
            "val": resp_dict["score"]
        }
    else: 
        return None 
    
def get_tempo_entailment(utt):
    tempo = "I'm in the mood for some music with a fast-paced tempo."

    llm_prompt = entailment_llm_prompt(tempo, utt)

    resp_json = get_llm_response(llm_prompt)
    resp_dict = json.loads(resp_json)

    if(resp_dict["score"] != 0.5):
        return {
            "attribute": "tempo",
            "match": False,
            "boolv": False,
            "phrase": None,
            "range": True,
            "val": resp_dict["score"]
        }
    else: 
        return None 

def get_valence_entailment(utt):
    valence = "I'm in the mood for some happy, cheerful music."

    llm_prompt = entailment_llm_prompt(valence, utt)

    resp_json = get_llm_response(llm_prompt)
    resp_dict = json.loads(resp_json)

    if(resp_dict["score"] != 0.5):
        return {
            "attribute": "valence",
            "match": False,
            "boolv": False,
            "phrase": None,
            "range": True,
            "val": resp_dict["score"]
        }
    else: 
        return None 
    
def get_attribute_numbers(utt):

    query = []

    dance = get_dancibility_entailment(utt)
    if(dance):
         query.append(dance)

    energy = get_energy_entailment(utt)
    if(energy):
        query.append(energy)

    loud = get_loudness_entailment(utt)
    if(loud):
        query.append(loud )

    acoustic = get_acousticness_entailment(utt)
    if(acoustic):
        query.append(acoustic)

    inst = get_instrumentalness_entailment(utt)
    if(inst):
        query.append(inst)

    live = get_liveness_entailment(utt) 
    if(live):
        query.append(live) 

    tempo = get_tempo_entailment(utt)
    if(tempo):
        query.append(tempo )

    valence = get_valence_entailment(utt)
    if(valence):
        query.append(valence)
    
    return query