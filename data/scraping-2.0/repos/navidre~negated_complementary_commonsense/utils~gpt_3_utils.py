import os
from pathlib import Path
from turtle import st
from dotenv import load_dotenv
import openai

HOME = str(Path.home())

TEMPERATURE = 0.7 # t
FREQUENCY_PENALTY = 0 # fp 
PRESENCE_PENALTY = 0 # pp
MAX_TOKENS = 250 # mt
TOP_P = 1 # tp
ENGINE = 'text-davinci-002' # e

load_dotenv(f'{Path().resolve()}/.env')
openai.api_key = os.environ['OPENAI_API_KEY']

# Few shot prompt
# TODO: Update normal to match negated
FEW_SHOT_PROMPT = "- PersonX accepts PersonX's diploma. As a result, others feel proud.\n- You are likely to find a basket in office.\n- Nose can be used to sense odors.\n- PersonX does PersonY's work. PersonX will be tired.\n- PersonX moves away. Before that, PersonX says goodbye to their friends."
NEGATED_FEW_SHOT_PROMPT = "- PersonX accepts PersonY's invitation. As a result, PersonY does not feel sad.\n- You are not likely to find car in house. \n- Hammer cannot be used for typing.\n- PersonX cuts PersonX. PersonX will not be happy.\n- PersonX runs. Before that, it is not needed that he bikes."

# Few shot prompts Q/A style
# FEW_SHOT_QA_PROMPT = "Q: PersonX accepts PersonX's diploma. As a result, what others feel? Name three.\nA: proud; jealous; joyful.\n\nQ: Where can you find a basket? Name three.\nA: an office; laundry room; closet.\n\nQ: What nose can be used for?\nA: sense odors; inhaling; exhaling.\n\nQ: PersonX does PersonY's work. What will be the effect on PersonX?\nA: tired; busy; overwhelmed.\n\nQ: PersonX moves away. What is done before that?\nA: PersonX says goodbye to their friends; Person X boxes belongings; PersonX researches the new place."
FEW_SHOT_QA_PROMPT = "Q: PersonX accepts PersonY's invitation. As a result, what PersonY feels? Name three.\nA: appreciated; happy; accepted.\n\nQ: Where can you find a car? Name three.\nA: road; garage; parking.\n\nQ: What hammer can be used for? Name three.\nA: breaking things; hitting a nail; carpentry.\n\nQ: PersonX cuts PersonX. What will PersonX feel? Name three.\nA: pain; sad; ill.\n\nQ: PersonX runs. Before that, what is needed? Name three.\nA: running attire; a place to run; legs."
NEGATED_FEW_SHOT_QA_PROMPT = "Q: PersonX accepts PersonY's invitation. As a result, what PersonY does not feel? Name three.\nA: sad; alone; rejected.\n\nQ: Where you cannot find a car? Name three.\nA: a living room; a nail salon; a bedroom.\n\nQ: What hammer cannot be used for? Name three.\nA: typing; toilet plunging; screwing a bolt.\n\nQ: PersonX cuts PersonX. What will PersonX not feel? Name three.\nA: happy; satisfied; relaxed.\n\nQ: PersonX runs. Before that, what is not needed? Name three.\nA: to bike; to jump; to eat."

# Few shot prompts Q/A style with chain-of-thought
# TODO: Update normal to match negated
# FEW_SHOT_COT_QA_PROMPT = "Q: PersonX accepts PersonX's diploma. As a result, what others feel? Name three.\nA: Gaining a diploma means that a person has put a lot of effort in a program an has now successfully finished it. The answers are: proud; jealous; joyful.\n\nQ: Where can you find a basket? Name three.\nA: A basket is usually used to hold things in and can be found where there things to hold. The answers are: an office; laundry room; closet.\n\nQ: What nose can be used for?\nA: Nose is an organ of the body, which is usually is for breathing and sensing odors. The answers are: sense odors; inhaling; exhaling.\n\nQ: PersonX does PersonY's work. What will be the effect on PersonX?\nA: A work is usually an energy-consuming task and anyone who does it has to spend time and energy. The answers are: tired; busy; overwhelmed.\n\nQ: PersonX moves away. What is done before that?\nA: Moving away means going from a primary place of living to another distant location, usually permanently. The answers are: PersonX says goodbye to their friends; Person X boxes belongings; PersonX researches the new place."
FEW_SHOT_COT_QA_PROMPT = "Q: PersonX accepts PersonY's invitation. As a result, what PersonY feels? Name three.\nA: By Accepting PersonX's invitation, PersonY intends to attend PersonX's event. Therefore, PersonY feels happy and appreciated. The answers are: appreciated; happy; accepted.\n\nQ: Where can you find a car? Name three.\nA: A car is a transportation tool. Therefore, it is usually driven in roads and sometimes off-road, depending on the vehicle. The answers are: road; garage; parking.\n\nQ: What hammer can be used for? Name three.\nA: Hammer is a tool to hit things hard with. Therefore, hitting nails or hard things. The answers are: breaking things; hitting a nail; carpentry.\n\nQ: PersonX cuts PersonX. What will PersonX feel? Name three.\nA: Cutting one's hand usually results in pain, bleeding, and sometimes infection. Therefore, PersonX will feel pain, discomfort, or ill. The answers are: pain; sad; ill.\n\nQ: PersonX runs. Before that, what is needed? Name three.\nA: Running is an act moving very fast on foot. Therefore, PersonX needs some running attire and shoes, and a vast place to run in. The answers are: running attire; a place to run; legs."
NEGATED_FEW_SHOT_COT_QA_PROMPT = "Q: PersonX accepts PersonY's invitation. As a result, what PersonY does not feel? Name three.\nA: By Accepting PersonX's invitation, PersonY intends to attend PersonX's event. The answers are: sad; alone; rejected.\n\nQ: Where you cannot find a car? Name three.\nA: Car is a transportation tool and usually driven in roads and sometimes off-road, depending on the vehicle. The answers are: a living room; a nail salon; a bedroom.\n\nQ: What hammer cannot be used for? Name three.\nA: Hammer is a tool to hit things hard with, such as nails. The answers are: typing; toilet plunging; screwing a bolt.\n\nQ: PersonX cuts PersonX. What will PersonX not feel? Name three.\nA: Cutting one's hand usually results in pain, bleeding, and sometimes infection. The answers are: happy; satisfied; relaxed.\n\nQ: PersonX runs. Before that, what is not needed? Name three.\nA: To run, PersonX needs some running attire and shoes. PersonX also needs a vast place to run in. The answers are: to bike; to jump; to eat."

# Few shot prompts Q/A style for updated_cot_qa
# TODO: Update normal to match negated
FEW_SHOT_UPDATED_COT_QA_PROMPT = FEW_SHOT_COT_QA_PROMPT
NEGATED_FEW_SHOT_UPDATED_COT_QA_PROMPT = "Q: PersonX accepts PersonY's invitation. As a result, what PersonY does not feel? Name three.\nA: By Accepting PersonX's invitation, PersonY intends to attend PersonX's event. If accepting, PersonY will be appreciated. So the answer should be opposite to this. The answers are: sad; alone; rejected.\n\nQ: Where you cannot find a car? Name three.\nA: Car is a transportation tool and usually driven in roads and sometimes off-road, depending on the vehicle. So the answer should be opposite to this. The answers are: a living room; a nail salon; a bedroom.\n\nQ: What hammer cannot be used for? Name three.\nA: Hammer is a tool to hit things hard with, such as nails. So the answer should be opposite to this. The answers are: typing; toilet plunging; screwing a bolt.\n\nQ: PersonX cuts PersonX. What will PersonX not feel? Name three.\nA: Cutting one's hand usually results in pain, bleeding, and sometimes infection. Cutting results in pain. So the answer should be opposite to this. The answers are: happy; satisfied; relaxed.\n\nQ: PersonX runs. Before that, what is not needed? Name three.\nA: To run, PersonX needs some running attire and shoes. PersonX also needs a vast place to run in. So the answer should be opposite to this. The answes are: to bike; to jump; to eat;"

# COT/QA with negation teaching
# TODO: Update normal to match negated
FEW_SHOT_COT_QA_NEG_TEACH_PROMPT = FEW_SHOT_COT_QA_PROMPT
NEGATED_FEW_SHOT_COT_QA_NEG_TEACH_PROMPT = "\"Not\" usually negates things. So make special attention to that.\n\nQ: PersonX accepts PersonY's invitation. As a result, what PersonY does not feel? Name three.\nA: By Accepting PersonX's invitation, PersonY intends to attend PersonX's event. Therfore, PersonY feels happy and appreciated. To answer \"does not\", you need to negate feeling of happiness and appreciation.  The answers are: sad; alone; rejected.\n\nQ: Where you cannot find a car? Name three.\nA: Car is a transportation tool and usually driven in roads and sometimes off-road, depending on the vehicle. To answer \"cannot\", you need to negate this answer. The answers are: a living room; a nail salon; a bedroom.\n\nQ: What hammer cannot be used for? Name three.\nA: Hammer is a tool to hit things hard with, such as nails. To answer \"cannot\", you need to negate nails. The answers are: typing; toilet plunging; screwing a bolt.\n\nQ: PersonX cuts PersonX. What will PersonX not feel? Name three.\nA: Cutting one's hand usually results in pain, bleeding, and sometimes infection. To answer \"not feel\", you need to negate these scenarios. The answers are: happy; satisfied; relaxed.\n\nQ: PersonX runs. Before that, what is not needed? Name three.\nA: To run, PersonX needs some running attire and shoes. PersonX also needs a vast place to run in. To answer \"not needed\", you need to negate the things that are needed. The answes are: to bike; to jump; to eat;\n\nQ: PersonX is wearing uniform and is in a cop car. Who PersonX cannot be?\nA: To answer, you need to negate the person that can be. In this case, it is cop. So the answer is: a civilian."

# COT/QA with updated negation teaching
# TODO: Update normal to match negated
FEW_SHOT_COT_QA_UPDATED_NEG_TEACH_PROMPT = FEW_SHOT_COT_QA_PROMPT
NEGATED_FEW_SHOT_COT_QA_UPDATED_NEG_TEACH_PROMPT = "\"Not\" usually negates things. So make special attention to that.\n\nQ: PersonX accepts PersonY's invitation. As a result, what PersonY does not feel? Name three.\nA: Let's first answer what PersonY feels if PersonX accepts PersonY's invitation. By Accepting PersonX's invitation, PersonY intends to attend PersonX's event. Therefore, PersonY feels happy and appreciated. To answer \"does not\", you need to negate the feeling of happiness and appreciation. The answers are: sad; alone; rejected.\n\nQ: Where you cannot find a car? Name three.\nA: Let's first answer where you can find a car. A car is a transportation tool. Therefore, it is usually driven in roads and sometimes off-road, depending on the vehicle. To answer \"cannot\", you need to negate this answer. The answers are: a living room; a nail salon; a bedroom.\n\nQ: What hammer cannot be used for? Name three.\nA: Let's first answer where hammer can be used. Hammer is a tool to hit things hard with. Therefore, hitting nails or hard things. To answer \"cannot be\", you need to negate nails. The answers are: typing; toilet plunging; screwing a bolt.\n\nQ: PersonX cuts PersonX. What will PersonX not feel? Name three.\nA: Let's first answer what will PersonX feel if cuts himself. Cutting one's hand usually results in pain, bleeding, and sometimes infection. Therefore, PersonX will feel pain, discomfort, or ill. To answer \" will not feel\", you need to negate these scenarios. The answers are: happy; satisfied; relaxed.\n\nQ: PersonX runs. Before that, what is not needed? Name three.\nA: Let's first answer what PersonX needs before PersonX runs. Running is an act moving very fast on foot. Therefore, PersonX needs some running attire and shoes, and a vast place to run in. To answer \"not needed\", you need to negate the things that are needed. The answers are: a car; a full stomach; a laptop."

# Ablation: COT/QA with negation teaching and variable temperature
FEW_SHOT_COT_QA_UPDATED_NEG_TEACH_PROMPT_ABLATED = FEW_SHOT_COT_QA_PROMPT
NEGATED_FEW_SHOT_COT_QA_UPDATED_NEG_TEACH_PROMPT_ABLATED = "Q: PersonX accepts PersonY's invitation. As a result, what PersonY does not feel? Name three.\nA: By Accepting PersonX's invitation, PersonY intends to attend PersonX's event. The answers are: sad; alone; rejected.\n\nQ: Where you cannot find a car? Name three.\nA: Cutting one's hand usually results in pain, bleeding, and sometimes infection. The answers are: happy; satisfied; relaxed.\n\nQ: What hammer cannot be used for? Name three.\nA: Hammer is a tool to hit things hard with. The answers are: typing; toilet plunging; screwing a bolt.\n\nQ: PersonX cuts PersonX. What will PersonX not feel? Name three.\nA: Cutting one's hand usually results in pain, bleeding, and sometimes infection. The answers are: happy; satisfied; relaxed.\n\nQ: PersonX runs. Before that, what is not needed? Name three.\nA: Running is an act moving very fast on foot. The answers are: a car; a full stomach; a laptop."

PROMPTS = {
    'few_shot': {'normal': FEW_SHOT_PROMPT, 'negated': NEGATED_FEW_SHOT_PROMPT},
    'few_shot_qa': {'normal': FEW_SHOT_QA_PROMPT, 'negated': NEGATED_FEW_SHOT_QA_PROMPT},
    'cot_qa': {'normal': FEW_SHOT_COT_QA_PROMPT, 'negated': NEGATED_FEW_SHOT_COT_QA_PROMPT},
    'updated_cot_qa': {'normal': FEW_SHOT_UPDATED_COT_QA_PROMPT, 'negated': NEGATED_FEW_SHOT_UPDATED_COT_QA_PROMPT},
    'cot_qa_neg_teach': {'normal': FEW_SHOT_COT_QA_NEG_TEACH_PROMPT, 'negated': NEGATED_FEW_SHOT_COT_QA_NEG_TEACH_PROMPT},
    'cot_qa_neg_teach_var_temp': {'normal': FEW_SHOT_COT_QA_NEG_TEACH_PROMPT, 'negated': NEGATED_FEW_SHOT_COT_QA_NEG_TEACH_PROMPT},
    'cot_qa_updated_neg_teach_var_temp': {'normal': FEW_SHOT_COT_QA_UPDATED_NEG_TEACH_PROMPT, 'negated': NEGATED_FEW_SHOT_COT_QA_UPDATED_NEG_TEACH_PROMPT},
    'cot_qa_updated_neg_teach_var_temp_ablated': {'normal': FEW_SHOT_COT_QA_UPDATED_NEG_TEACH_PROMPT_ABLATED, 'negated': NEGATED_FEW_SHOT_COT_QA_UPDATED_NEG_TEACH_PROMPT_ABLATED}
}

# Predicate is key.
# n is the number of items that we are looking for.
# pred: (find_subj, find_obj)
QUESTION_TEMPLATES = {'AtLocation': {'normal': 'Where is the {subj} located? Name {n}.', 'negated': 'Where is the {subj} not located? Name {n}.'},
                      'CapableOf': {'normal': 'What is {subj} capable of? Name {n}.', 'negated': 'What is {subj} not capable of? Name {n}.'},
                      'Causes': {'normal': 'What {subj} can cause? Name {n}.', 'negated': 'What {subj} cannot cause? Name {n}.'},
                      'CausesDesire': {'normal': 'What {subj} causes desire in? Name {n}', 'negated': 'What {subj} does not cause desire in? Name {n}.'},
                      'CreatedBy': {'normal': 'What is created by {subj}? Name {n}.', 'negated': 'What is not created by {subj}? Name {n}.'},
                      'DefinedAs': {'normal': 'What is the definition of {subj}? Name {n}.', 'negated': 'What is not the definition of {subj}? Name {n}.'},
                      'Desires': {'normal': 'What does {subj} desire to do? Name {n}.', 'negated': 'What does {subj} not desire to do? Name {n}.'},
                      'HasA': {'normal': 'What does {subj} have? Name {n}.', 'negated': 'What does {subj} not have? Name {n}.'},
                      'HasFirstSubevent': {'normal': 'What happens when you {subj}? Name {n}.', 'negated': 'What does not happen when you {subj}? Name {n}.'},
                      'HasPrerequisite': {'normal': 'What does {subj} have as a prerequisite? Name {n}.', 'negated': 'What does {subj} not have as a prerequisite? Name {n}.'},
                      'HasProperty': '',
                      'IsA': '',
                      'MadeOf': '',
                      'MotivatedByGoal': '',
                      'NotCapableOf': '',
                      'NotHasProperty': '',
                      'NotIsA': '',
                      'PartOf': '',
                      'ReceivesAction': '',
                      'RelatedTo': '',
                      'UsedFor': '',
                      # ATOMIC-2020
                      'xWant': {'normal': '{subj}. What does PersonX want to do? Name {n}.', 'negated': '{subj}. What does PersonX not want to do? Name {n}.'},
                      'xReact': {'normal': '{subj}. What does PersonX feel about it? Name {n}.', 'negated': '{subj}. What does PersonX not feel about it? Name {n}.'}, 
                      'oWant': {'normal': '{subj}. What does PersonY want to do? Name {n}.', 'negated': '{subj}. What does PersonY not want to do? Name {n}.'},
                      'HinderedBy': {'normal': '{subj}. What can hinder/obstruct it? Name {n}.', 'negated': '{subj}. What cannot hinder/obstruct it? Name {n}.'},
                      'isBefore': {'normal': '{subj}. What happens before it? Name {n}.', 'negated': '{subj}. What does not happen before it? Name {n}.'},
                      'isAfter': {'normal': '{subj}. What happens after it? Name {n}.', 'negated': '{subj}. What does not happen after it? Name {n}.'},
                      'HasSubEvent': {'normal': '{subj}. What will you do while: {subj}? Name {n}.', 'negated': '{subj}. What you will not do while: {subj}? Name {n}.'},
                      # Negated-CS
                      'Is': {'normal': 'What is the concept (denoted by X) described here? Name {n}.', 'negated': 'What is not the concept (denoted by X) described here? Name {n}.'},
                      }

SELF_EVALUATE_PROMPT = "Please evaluate each question/answer pair with only yes or no. Each question is related to commonsense. yes if the answer is correct and no if the answer is incorrect. Please note that all questions are negated complementary and use cannot, does not, etc.\n\nQ: Question: What fish cannot do? Answer: run. Is this answer correct? Reply with yes or no.\nA: yes\n\nQ: Question: What zebra cannot do? Answer: run. Is this answer correct? Reply with yes or no.\nA: no\n\nQ: Question: PersonX accepts PersonY's invitation. As a result, what PersonY does not feel? Answer: rejected. Is this answer correct? Reply with yes or no.\nA: yes\n\nQ: Question: PersonX accepts PersonY's invitation. As a result, what PersonY does not feel? Answer: happy. Is this answer correct? Reply with yes or no.\nA: no\n\nQ: Question: PersonX cuts PersonX. What will PersonX not feel? Answer: relaxed. Is this answer correct? Reply with yes or no.\nA: yes"
NORMAL_SELF_EVALUATE_PROMPT = "Please evaluate each question/answer pair with only yes or no. Each question is related to commonsense. yes if the answer is correct and no if the answer is incorrect.\n\nQ: Question: What fish can do? Answer: run. Is this answer correct? Reply with yes or no.\nA: no\n\nQ: Question: What zebra can do? Answer: run. Is this answer correct? Reply with yes or no.\nA: yes\n\nQ: Question: PersonX accepts PersonY's invitation. As a result, what does PersonY feel? Answer: rejected. Is this answer correct? Reply with yes or no.\nA: no\n\nQ: Question: PersonX accepts PersonY's invitation. As a result, what does PersonY feel? Answer: happy. Is this answer correct? Reply with yes or no.\nA: yes\n\nQ: Question: PersonX cuts PersonX. What will PersonX feel? Answer: relaxed. Is this answer correct? Reply with yes or no.\nA: no"

NUMBER_TO_TEXT = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}

def generate_zero_shot_using_gpt_3(prompt:str, temperature:float=TEMPERATURE, max_tokens:int=MAX_TOKENS, top_p:float=TOP_P, frequency_penalty:float=FREQUENCY_PENALTY, presence_penalty:float=PRESENCE_PENALTY, engine:str=ENGINE):
    """ Generate a zero-shot response using GPT-3.

    Args:
        prompt (str): The prompt to use with the model.
        temperature (float, optional): Temperature. Defaults to TEMPERATURE.
        max_tokens (int, optional): Number of max tokens generated in the output. Defaults to MAX_TOKENS.
        top_p (float, optional): Nucleus sampling. Top possible tokens with cumulative probability of at least top_p. Defaults to TOP_P.
        frequency_penalty (float, optional): frequency_penalty. Defaults to FREQUENCY_PENALTY.
        presence_penalty (float, optional): presence_penalty. Defaults to PRESENCE_PENALTY.
        engine (str, optional): Name of the GPT-3 engine to use. Defaults to ENGINE.

    Returns:
        (str, str): The generated text answer and the overall response.
    """
    start_sequence = " "
    prompt_as_input = f"{prompt}{start_sequence}"
    response = openai.Completion.create(
                model=engine,
                prompt=prompt_as_input,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=presence_penalty,
                presence_penalty=frequency_penalty,
                logprobs=20
                )
    text_answer = response['choices'][0]['text']
    return text_answer, response

def generate_few_shot_using_gpt_3(few_shot_prompt:str, premise:str, temperature:float=TEMPERATURE, max_tokens:int=MAX_TOKENS, top_p:float=TOP_P, frequency_penalty:float=FREQUENCY_PENALTY, presence_penalty:float=PRESENCE_PENALTY, engine:str=ENGINE):
    """ Generate a zero-shot response using GPT-3.

    Args:
        prompt (str): The few-shot example prompt to use with the model.
        premise (str): The premise to use with the model. Verbalized subject + predicate.
        temperature (float, optional): Temperature. Defaults to TEMPERATURE.
        max_tokens (int, optional): Number of max tokens generated in the output. Defaults to MAX_TOKENS.
        top_p (float, optional): Nucleus sampling. Top possible tokens with cumulative probability of at least top_p. Defaults to TOP_P.
        frequency_penalty (float, optional): frequency_penalty. Defaults to FREQUENCY_PENALTY.
        presence_penalty (float, optional): presence_penalty. Defaults to PRESENCE_PENALTY.
        engine (str, optional): Name of the GPT-3 engine to use. Defaults to ENGINE.

    Returns:
        (str, str): The generated text answer and the overall response.
    """
    start_sequence = "- "
    prompt_as_input = f"{few_shot_prompt}\n{start_sequence}{premise}"
    response = openai.Completion.create(
                model=engine,
                prompt=prompt_as_input,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logprobs=20,
                stop=[". ", "\n"]
                )
    text_answer = response['choices'][0]['text']
    return text_answer, response

def generate_few_shot_qa(few_shot_prompt:str, question:str, temperature:float=TEMPERATURE, max_tokens:int=MAX_TOKENS, top_p:float=TOP_P, frequency_penalty:float=FREQUENCY_PENALTY, presence_penalty:float=PRESENCE_PENALTY, engine:str=ENGINE):
    """ Generate a few-shot Q/A-based response using GPT-3.

    Args:
        prompt (str): The few-shot example prompt to use with the model.
        question (str): The question to ask.
        temperature (float, optional): Temperature. Defaults to TEMPERATURE.
        max_tokens (int, optional): Number of max tokens generated in the output. Defaults to MAX_TOKENS.
        top_p (float, optional): Nucleus sampling. Top possible tokens with cumulative probability of at least top_p. Defaults to TOP_P.
        frequency_penalty (float, optional): frequency_penalty. Defaults to FREQUENCY_PENALTY.
        presence_penalty (float, optional): presence_penalty. Defaults to PRESENCE_PENALTY.
        engine (str, optional): Name of the GPT-3 engine to use. Defaults to ENGINE.

    Returns:
        (str, str): The generated text answer and the overall response.
    """
    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "
    prompt_as_input = f"{few_shot_prompt}{restart_sequence}{question}{start_sequence}"
    response = openai.Completion.create(
                model=engine,
                prompt=prompt_as_input,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=presence_penalty,
                presence_penalty=frequency_penalty,
                logprobs=20,
                stop=["\n"]
                )
    text_answer = response['choices'][0]['text']
    return text_answer, response

# TODO: Update this for the CSQA dataset
def q_and_a_gpt3(prompt:str, question:str, temperature:float=TEMPERATURE, max_tokens:int=MAX_TOKENS, top_p:float=TOP_P, frequency_penalty:float=FREQUENCY_PENALTY, presence_penalty:float=PRESENCE_PENALTY, engine:str=ENGINE):
    """ Generic method for generating a response to a question. Q/A style of prompting.

    Args:
        prompt (str): The static part of the prompt to use with the model.
        question (str): The question to ask the model. Technically part of prompt.
        temperature (float, optional): Temperature. Defaults to TEMPERATURE.
        max_tokens (int, optional): Number of max tokens generated in the output. Defaults to MAX_TOKENS.
        top_p (float, optional): Nucleus sampling. Top possible tokens with cumulative probability of at least top_p. Defaults to TOP_P.
        frequency_penalty (float, optional): frequency_penalty. Defaults to FREQUENCY_PENALTY.
        presence_penalty (float, optional): presence_penalty. Defaults to PRESENCE_PENALTY.
        engine (str, optional): Name of the GPT-3 engine to use. Defaults to ENGINE.

    Returns:
        (str, str): The generated text answer and the overall response.
    """
    start_sequence = "\nA:"
    restart_sequence = "\n\nQ: "
    prompt_as_input = f"{prompt}{restart_sequence}{question}{start_sequence}"
    response = openai.Completion.create(
                model=engine,
                prompt=prompt_as_input,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logprobs=20,
                stop=["\n"]
                )
    text_answer = response['choices'][0]['text']
    return text_answer, response