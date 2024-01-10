import os
import json
from langchain.llms import LlamaCpp
from colorama import Fore as c

def load_model():
    '''
    Method to load the model. Gets the model name from config.json.
    '''
    
    print(f'\n[ {c.CYAN}MODEL{c.RESET} ] Loading model...\n')

    # Read model name from config.json
    with open('data/config.json', 'r') as f:
        config = json.load(f)

    model_name = config['SelectedModel']
    model_config = config[model_name]

    model_path = f'./models/{model_config["model"]}'
    n_ctx = model_config['n_ctx']
    temperature = model_config['temperature']
    max_tokens = model_config['max_tokens']
    top_p = model_config['top_p']
    top_k = model_config['top_k']
    verbose = model_config['verbose']
    template = model_config['template']
    repeat_penalty = model_config['repeat_penalty']

    llm = LlamaCpp(
        model_path=model_path, 
        n_ctx=n_ctx,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        verbose=verbose,
        repeat_penalty=repeat_penalty,
        echo=True
        )
    
    print(f'\n[ {c.GREEN}MODEL{c.RESET} ] Model loaded\n')

    return llm, template

def get_models():
    '''
    Method to get the models available in the models folder.
    '''
    # Check if models folder exists
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    # Check files in models folder
    print(f'\n[ {c.CYAN}MODEL{c.RESET} ] Getting models...')
    models = []
    for file in os.listdir('./models'):
        models.append(file)
    print(f'[ {c.GREEN}MODEL{c.RESET} ] Models obtained')

    return models

def set_model(model_config_name:str, model_name:str): 
    '''
    Method to set the model. Where:
        - model_name: name of the model. You can get the models with get_models()
    '''

    print(f'\n[ {c.CYAN}MODEL{c.RESET} ] Setting up model {model_name}')
    try:
        models = get_models()
        # Check if model exists
        if model_name not in models:
            print(f'\n[ {c.RED}MODEL{c.RESET} ] Model {model_name} does not exist')
            return f'Model {model_name} does not exist'
        
        # Write the model name in config.json
        with open('data/config.json', 'r') as f:
            config = json.load(f)

        config_models = list(config.keys())
        config_models.pop(0)

        if model_config_name in config_models:
            config['SelectedModel'] = model_config_name
        else:
            print(f'\n[ {c.RED}MODEL{c.RESET} ] Model {model_config_name} is not configured in config.json')
            return f'Model {model_config_name} not configured in config.json'
        
        config[model_config_name]['model'] = model_name
        with open('data/config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f'\n[ {c.GREEN}MODEL{c.RESET} ] Model {model_name} set up correctly')

        return True
    except Exception as e:
        print(e)
        return False