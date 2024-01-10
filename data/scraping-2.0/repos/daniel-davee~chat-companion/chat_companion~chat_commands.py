from typing import Union,Optional
from pathlib import Path
from json import loads,dumps
from os import environ
from InquirerPy.inquirer import text, fuzzy, select
from toolbox.companion_logger import logger
from toolbox.prompt_tools import clean_f_str
import openai
import shelve

cwd = Path(__file__).parent.parent
# cwd = Path().cwd()

contexts = cwd / ".contexts/contexts.dat"
contexts.parent.mkdir(exist_ok=True)
contexts = '.'.join(contexts.as_posix().split('.')[:-1])

logs = cwd / "log"
logs.mkdir(exist_ok=True)

def get_default_model()->str:
    settings_file:Path = cwd / 'settings.json'
    return loads(settings_file.read_text())['models']['default']

def set_default_model(value:str,key:str='default'):
    settings_file:Path = cwd / 'settings.json'
    print(f'Changing {key} to {value}')
    settings:dict[str,str] = loads(settings_file.read_text())
    settings['models'][key] = value
    settings_file.write_text(dumps(settings))

def proof_read(prompt:str,temperature:float = 0.5,) -> str:
    """
    Proof reads a prompt
    
    Args:
        prompt (str)The prompt you want to proof read
        temperature (float, optional): for more random. Defaults to 0.5.
    Returns:
        str: corrected prompt
    """
    
    prompt = clean_f_str(f'''
                         Proof read this '{prompt or 
                         text('What do you want to proof read?')
                         .execute()}',
                         and correct any spelling or grammar mistakes.
                        ''')
    return generate_response(prompt,temperature) 

def translate(prompt:str,temperature:float = 0.5,language:str='english',) -> str:
    """
    Translates a prompt    
    
    Args:
        prompt (str): _description_
        temperature (float, optional): _description_. Defaults to 0.5.
        language (str, optional): _description_. Defaults to 'english'.
    Returns:
        str: translated prompt
    """
    
    prompt = clean_f_str(f'''Translate this '{prompt or 
                             text('What do you want to translate?')
                             .execute()}' into {language}''')
    return generate_response(prompt,temperature) 

def get_key()->str:
    if 'CHATKEY' not in environ: raise Exception('Set CHATKEY')
    return environ['CHATKEY']
def generate_response( prompt:str='', temperature:float = 0.5,
                      engine:Optional[str]=None, max_tokens:int=1024,
                      n:int =1,filename:str='', bulk:bool=False,
                      )->Union[str,list[str]]:
    """
    This Generates a response, it doesn't store it context.db.
    
    Args:
        prompt (str):Your prompt
        temperature (float):1 for more random. Defaults to 0.5.
        engine (str):The engine you use davinci|curie|ada. Defaults to 'davinci'.
        max_tokens (int):max tokens used in response. Defaults to 1024.
        n (int):The number of generated responses Defaults to 1.
        filename (str):The file name to output review for example scratch.py. Defaults to ''.
        bulk (bool):If set will return all responses as a list. Defaults to False.

    Returns:
        str or List[str]: the response or list of responses
    """
     
    openai.api_key = get_key()
    engine:str = engine or get_default_model()
    completions = openai.Completion.create(
                                    engine=engine,
                                    prompt=prompt,
                                    max_tokens=max_tokens,
                                    n=n,
                                    stop=None,
                                    temperature=temperature,
                                    )
    choices = [c.text for c in completions.choices]
    response = choices if bulk else \
               select('Choose Response',choices=choices ).execute() if n > 1 \
               else completions.choices[0].text
    
    if filename:
        (Path()/filename).write_text(response)
    
    return response

def summarize( prompt:str,temperature:float = 0.5,
              engine:Optional[str]=None, max_tokens:int=3000,
              n:int =1,filename:str='',bulk:bool=False,type_:str='',
              ) -> str:
    """
    Summarizes input

    Args:
        prompt (str):Your prompt. Defaults to ''.
        engine (str):The engine you use davinci|curie|ada. Defaults to 'davinci'.
        n (int):The number of generated responses. Defaults to 1.
        temperature (float):1 for more random. Defaults to 0.5.
        type_ (str):The type of thing you are summarizing, e.g. a conversation. Defaults to ''.

    Returns:
        str: the summary
    """
    
    return generate_response(
                          prompt=f'Summarize this {type_} {prompt} concisely',
                          engine=engine,max_tokens=max_tokens,n=n,
                          temperature=temperature, filename=filename,bulk=bulk,)
    
    
def talk(prompt:str = '',temperature:float = 0.5,
         engine:Optional[str]=None,max_tokens:int=1024,n:int =1,
         filename:str='',profile:str='default',in_file:str='',**kwargs,
        )->str:
    """
    This allows you save your companion's responses, they are stored in context.db.

    Args:
        prompt (str):Your prompt
        temperature (float):1 for more random. Defaults to 0.5.
        engine (str):The engine you use davinci|curie|ada. Defaults to 'davinci'.
        max_tokens (int):max tokens used in response. Defaults to 1024.
        n (int):The number of generated responses Defaults to 1.
        filename (str):The file name to output review for example scratch.py. Defaults to ''.
        profile (str):The profile to load in from contexts. Defaults to 'default'.
        in_file (str):File that is input to the prompt use. Defaults to ''.

    Returns:
        str: the response
    """
       
    prompt = prompt or text('What do you want to ask?').execute()
    if in_file or kwargs:
        logger.debug(f'{kwargs=}, {in_file=}')
        if in_file: kwargs['in_file'] = Path(in_file).read_text()
        prompt = [s.split('{') for s in prompt.split('}') if s]
        end =prompt[-1][0] if len(prompt[-1]) < 2 else ''
        prompt = [s for s in prompt if len(s) ==2]
        prompt = ''.join([f'{k}{kwargs[v]}' for k,v in prompt])+end
    
    logger.prompt(prompt)
    response=generate_response(
                  prompt=prompt,
                  engine=engine,
                  max_tokens=max_tokens,
                  n=n,
                  temperature=temperature,
                  filename=filename)
    logger.response(response)
    # logger.summary((summary := summarize(
    #                             prompt=f'you said to me "{prompt}". and I responded back to you with "{response}"',
    #                             t='conversations between you and me',
    #                             engine=engine,
    #                             n=n,
    #                             temperature=temperature,
    #                 )))
    # logger.debug(f'{summary=}')
     
    # with shelve.open(contexts, writeback=True) as hst:
    #     if profile not in hst: hst[profile] = {'history':{}}
    #     hst[profile]['history'] |= {prompt:{'response':response,'summary':summary}}
    
    return response


def review(filename:str='',profile:str='default',summary:bool=False,)->str:
    """
    To review previous questions and responses,
    use the `review` subcommand. This will bring up a list of previous questions.
    You can then select a question to view the response.

    Args:
        filename (str):The file name to output review for example scratch.py. Defaults to ''.
        profile (str):The profile to load in from contexts. Defaults to 'default'.
        summary (bool):show summary. Defaults to False.

    Returns:
        str: the previous response
    """
           
    with shelve.open(contexts, writeback=True) as hst:
        if profile not in hst: 
            hst[profile] = {'history':{}}
            return 'No history yet'
        prompt = fuzzy('What prompt do you want to review', 
                        choices=list(hst[profile]['history'].keys()),
                        vi_mode=True,
                        ).execute()
        response = hst[profile]['history'][prompt]['summary' if summary else 'response']
    if filename:(Path()/filename).write_text(response)
    return response


def resummarize(temperature:float = 0.5,n:int =1,profile:str='default',
                )->dict[str,dict[str,str]]:
    """
    Creates an updated summary for question.

    Args:
        profile (str):The profile to load in from contexts. Defaults to 'default'.
        temperature (float):1 for more random. Defaults to 0.5.
        n (int):The number of generated responses Defaults to 1.

    Returns:
        Dict[str,Dict[str,str]]: _description_
    """
        
    with shelve.open(contexts, writeback=True) as hst:
        if profile not in hst: 
            hst[profile] = {'history':{}}
            return 'No history yet'
        logger.prompt((prompt:=fuzzy('What prompt do you want to review', 
                        choices=list(hst[profile]['history'].keys()),
                        vi_mode=True,
                        ).execute()))
        logger.response((response:=hst[profile]['history'][prompt]['response']))
        logger.summary((summary := summarize(
                                    prompt=f'you said to me "{prompt}". and I responded back to you with "{response}"',
                                    t='conversations between you and me',
                                    n=n,
                                    temperature=temperature,
                        )))
        hst[profile]['history'][prompt]= {'response':response, 'summary':summary}
        return summary