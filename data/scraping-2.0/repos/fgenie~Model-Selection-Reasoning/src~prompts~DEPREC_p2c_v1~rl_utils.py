from typing import Sequence, Tuple, Dict, Union
import openai
import yaml

yaml.full_load()

KEY = open('/Users/seonils/dev/llm-reasoners/examples/Model-Selection-Reasoning/openai_key.txt').read().strip()
openai.api_key = KEY # set key

def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def sample_a_model(
            question:str='',
            models_to_sample:Sequence[str], 
            howtosample:str='random',
            **kwargs)->Dict[str, Union[str, Sequence[str]]]]:
    '''
    From `models_to_sample`, sample a `model(sampled)` to tackle a `question`

    !!!system_msg need to clarify the agent is trying to choose a good method to solve the question.!!!
    '''
    assert question.strip(), f"question need to be given."

    if howtosample == 'random':
        sampled = random.choice(models_to_sample)
        verbal_response = f"Q: {question}\nRandomly choose a method from: {list(set(models_to_sample))}.\nChoice: {sampled}"
    elif howtosample == 'llm':
        if backbone == 'chatgpt':
            model_name = 'gpt-3.5-turbo'
        elif backbone == 'gpt-4':
            model_name = backbone
        
        _pref = f"Q: {question}\nFrom {list(set(models_to_sample))}, which is most likely to be the best method to tackle the problem?"
        _verbal_response = query_modelsample(data, key=key, models_to_sample)
        sampled = parse_choice(verbal_response)
        if sampled not in models_to_sample:
            print(f'sampling failed:\n{_verbal_response}')
            sampled = random.choice(models_to_sample)
            _verbal_response = f"{_verbal_response}\n\n\nSampling failed. Randomly choose a method from: {list(set(models_to_sample))}.\nChoice: {sampled}"
    models_to_sample.remove(sampled)
    return {
        'sampled':sampled,
        'verbal_response':verbal_response,
        'models_to_sample':models_to_sample,
    }

def parse_choice(verbal_response:str)->str:
    lines = verbal_response.split('\n')
    for l in lines:
        if l.startswith('Choice: '):
            gist = l
            break
    return l.split('Choice: ')[-1]


def query_modelsample()->Tuple[str, str]:
    msgs = get_modelsample_prompt(data, models_to_sample)
    advice = completion_with_backoff(
            api_key=key,
            model= model_name,
            max_tokens=500,
            stop='\n\n\n',
            messages=query_message,
            temperature=cot_temperature,
            top_p=1.0,
            n=1)
    
    model_name = parse_model_name(advice['choices'][0]['message']['content'])
    return model_name, advice

def get_model_sample_prompt(data:Dict, models_to_sample:Sequence[str])->str:
    raise NotImplementedError('rl_utils.py::get_model_sample_prompt'')
    return 

