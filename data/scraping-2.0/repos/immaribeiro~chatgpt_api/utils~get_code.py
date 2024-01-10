import openai
import os
import json
import time

from utils.logger import get_logger

def get_code(prompt=None, number=None, model=None, engine=None, max_tokens=None, temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None, stop=None):
    logger = get_logger(__name__, 'utils/get_code.log')
    try:
        os.makedirs('output/code/', exist_ok=True)
        os.makedirs('utils/configs/code', exist_ok=True)

        # Load default values from json file
        with open('utils/configs/code/get_code_config.json') as f:
            config = json.load(f)
        
        # Define default values and update with any non-None arguments
        args = {
            'prompt': prompt or config.get('prompt'),
            'n': number or config.get('number'),
            'engine': engine or config.get('engine'),
            'model': model or config.get('model'),
            'max_tokens': max_tokens or config.get('max_tokens'),
            'temperature': temperature or config.get('temperature'),
            'top_p': top_p or config.get('top_p'),
            'frequency_penalty': frequency_penalty or config.get('frequency_penalty'),
            'presence_penalty': presence_penalty or config.get('presence_penalty'),
            'stop': stop or config.get('stop'),
        }
        
        # Raise an error if both engine and argument are provided
        if engine is not None and model is not None:
            raise ValueError("Only one of 'engine' or 'model' can be provided.")
        
        # Remove any arguments that are still None
        args = {k: v for k, v in args.items() if v is not None}

        response = openai.Completion.create(**args)
        #selecting only the first result to print
        code = response.choices[0].text

        #writing to file all the results
        current_time = str(int(time.time()))
        prompt_short = prompt[:min(len(prompt), 30)]
        filename = prompt_short.replace(" ", "_") + '_' + current_time +'.json'
        with open('output/code/' + filename, 'wb') as f:
                f.write(json.dumps(response).encode())
        logger.info('New code request made successfuly and saved in file ' + filename)
        
        time.sleep(0.1)
        return code

    except Exception as e:
        logger.debug(f'An error occurred while generating code: {str(e)}')
