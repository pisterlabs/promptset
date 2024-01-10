import openai
import configparser
import logging

import tqdm as td
import os, glob
from datetime import date
import time


def run_ner(output_dir: str, few_shot: bool = True, api_retry: int = 6):
    '''
    Do named entity recognition - problem, test, treatment
    
    output_dir should contain input data for the task.
    Recommend to execute generate_data.py before executing API functions.
    '''
    ### Get prompt parameters
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), "api.config"))
    
    openai.api_key = config['openai']['api_key']
    model = config['openai']['model']
    temp = float(config['openai']['temperature'])
    
    # Create folder to store output
    if few_shot:
        date_path = "output_" + "one_" + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/ner"
    else:
        date_path = "output_" + "zero_" + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/ner"
    path = os.path.join(output_dir, date_path)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Read NER input data
    notes = glob.glob(os.path.join(output_dir, 'data/ner', '*.txt'))
    
    ### Get prompt design
    if few_shot == True:
        system_msg = config['NER']['few_prompt']
        few_user = config['RE']['few_user']
        few_assistant = config['NER']['few_assistant']
        
        logging.info(f'start API reqeusts...')
        for note in td.tqdm(notes, desc = "Generating NER output from i2b2", unit = "files"):
            if os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')):
                logging.info('output exists: %s' % note)
            else:
                with open(note, 'r') as f:
                    content = f.read()
                
                # Call GPT API -> Re-call API upto 5 times
                api_no = 1
                while api_no < api_retry:
                    try:
                        if not api_no == 1:
                            logging.info(f'{api_no}th API re-requests...')
                        completions = openai.ChatCompletion.create(
                            model = model,
                            temperature = temp,
                            n = 1,
                            messages = [
                                {'role':'system', 'content':system_msg},
                                {'role':'user', 'content':few_user},
                                {'role':'assistant', 'content':few_assistant},
                                {'role':'user', 'content':content}
                            ]
                        )
                        response = completions.choices[0]['message']['content']
                        break
                    except Exception as e:
                        logging.error(f"{note}: {api_no}th API error: \n{e}")
                        logging.info(f"susepnding 30 secs to avoid max retries...\n")
                        time.sleep(30)
                        api_no += 1
                        response = ''
                        
                if not response=='':
                    # Remove incomplete reponse
                    lines = response.strip().split('\n')
                    lines = [line for line in lines if all(keyword in line for keyword in ('text', 'type'))]
                    response = '\n'.join(lines)
                    response = '<TAGS>\n' + response + '\n</TAGS>'
                    
                    output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(response)
                else: 
                    logging.info(f"pass saving {os.path.splitext(os.path.basename(note))[0]} file due to empty response...\n")
                    
    else: # zero_shot
        system_msg = config['NER']['zero_prompt']
        
        logging.info(f'start API reqeusts...')
        for note in td.tqdm(notes, desc = "Generating NER output from i2b2", unit = "files"):
            if os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')):
                logging.info('output exists: %s' % note)
            else:
                with open(note, 'r') as f:
                    content = f.read()
                
                api_no = 1
                while api_no < api_retry:    
                    try:
                        if not api_no == 1:
                            logging.info(f'{api_no}th API re-requests...')
                        completions = openai.ChatCompletion.create(
                            model = model,
                            temperature = temp,
                            n = 1,
                            messages = [
                                {'role':'system', 'content': system_msg},
                                {'role':'user', 'content':content}
                            ]
                        )
                        response = completions.choices[0]['message']['content']
                        break
                    except Exception as e:
                        logging.error(f"{note}: {api_no}th API error: \n{e}")
                        logging.info(f"susepnding 30 secs to avoid max retries...\n")
                        time.sleep(30)
                        api_no += 1
                        response = ''
                
                if not response == '':
                    lines = response.strip().split('\n')
                    lines = [line for line in lines if all(keyword in line for keyword in ('text', 'type'))]
                    response = '\n'.join(lines)
                    response = '<TAGS>\n' + response + '\n</TAGS>'
                    
                    output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
                    with open(output_file, 'w', encoding = 'utf-8') as f:
                        f.write(response)
                else: 
                    logging.ingo(f"pass saving {os.path.splitext(os.path.basename(note))[0]} file due to empty response...\n")
    

def run_re(output_dir: str, few_shot: bool = True, api_retry: int = 6):
    '''
    Do temporal relation extraction
    
    output_dir should contain input data for the task.
    Recommend to execute generate_data.py before executing API functions.
    '''
    ### Get prompt parameters
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), "api.config"))
    
    openai.api_key = config['openai']['api_key']
    model = config['openai']['model']
    temp = float(config['openai']['temperature'])
    
    # Create folder to store output
    if few_shot:
        date_path = "output_" + 'one_' + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/re"
    else:
        date_path = "output_" + 'zero_' + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/re"
    path = os.path.join(output_dir, date_path)
    if not os.path.exists(path):
        os.makedirs(path)
    # Read RE input data
    notes = glob.glob(os.path.join(output_dir, 'data/re', '*.txt'))
    
    ### Get prompt design
    if few_shot == True:
        system_msg = config['RE']['few_prompt']
        few_user = config['RE']['few_user']
        few_assistant = config['RE']['few_assistant']
        
        logging.info(f'start API requests...')
        for note in td.tqdm(notes, desc="Generating RE output from i2b2", unit="files"):
            if os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')):
                logging.info('output exists: %s' % note)
            else:
                with open(note, 'r') as f:
                    content = f.read()
                
                api_no = 1
                while api_no < api_retry:
                    try:
                        if not api_no == 1:
                            logging.info(f'{api_no}th API re-requests...')
                        completions = openai.ChatCompletion.create(
                            model = model,
                            temperature = temp,
                            n = 1,
                            messages = [
                                {'role':'system', 'content':system_msg},
                                {'role':'user', 'content':few_user},
                                {'role':'assistant', 'content':few_assistant},
                                {'role':'user', 'content':content}
                            ]
                        )
                        response = completions.choices[0]['message']['content']
                        break
                    except Exception as e:
                        logging.error(f"{note}: {api_no}th API error: \n{e}")
                        logging.info(f"susepnding 30 secs to avoid max retries...\n")
                        time.sleep(30)
                        api_no += 1
                        response = ''
                                            
                if not response == '':
                    # Remove the last XML entity if it doesn't have toID, fromID, or type.
                    lines = response.strip().split('\n')
                    lines = [line for line in lines if all(keyword in line for keyword in ('toID', 'fromID', 'type'))]
                    response = '\n'.join(lines)
                    response = '<TAGS>\n' + response + '\n</TAGS>'
                    
                    output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(response)
                else: 
                    logging.info(f"pass saving {os.path.splitext(os.path.basename(note))[0]} file due to empty response...\n")
        
    else:
        system_msg = config['RE']['zero_prompt']
        
        logging.info(f'start API requests...')
        for note in td.tqdm(notes, desc="Generating RE output from i2b2", unit="files"):
            if os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')):
                logging.info('output exists: %s' % note)
            else:
                with open(note, 'r') as f:
                    content = f.read()
                
                # GPT API call
                api_no = 1
                while api_no < api_retry:
                    try:
                        if not api_no == 1:
                            logging.info(f'{api_no}th API re-requests...')
                        completions = openai.ChatCompletion.create(
                            model = model,
                            temperature = temp,
                            n = 1,
                            messages = [
                                {'role':'system', 'content':system_msg},
                                {'role':'user', 'content':content}
                            ]
                        )
                        response = completions.choices[0]['message']['content']
                        break
                    except Exception as e:
                        logging.error(f"{note}: {api_no}th API error: \n{e}")
                        logging.info(f"susepnding 30 secs to avoid max retries...\n")
                        time.sleep(30)
                        api_no += 1
                        response = ''
                                        
                if not response == '':
                    lines = response.strip().split('\n')
                    lines = [line for line in lines if all(keyword in line for keyword in ('toID', 'fromID', 'type'))]
                    response = '\n'.join(lines)
                    response = '<TAGS>\n' + response + '\n</TAGS>'
                    
                    output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(response)
                else: 
                    logging.info(f"pass saving {os.path.splitext(os.path.basename(note))[0]} file...")

def run_nerre(output_dir: str, few_shot: bool = True, api_retry: int = 6):
    '''
    Do end-to-end relation extraction
    
    output_dir should contain input data for the task.
    Recommend to execute generate_data.py before executing API functions.
    '''
    ### Get prompt parameters
    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), "api.config"))
    
    openai.api_key = config['openai']['api_key']
    model = config['openai']['model']
    temp = float(config['openai']['temperature'])
    
    # Create folder to store output
    if few_shot:
        date_path = "output_" + 'one_' + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/nerre"
    else:
        date_path = "output_" + 'zero_' + config['openai']['model'] + '_' + date.today().strftime("%y%m%d") + "/nerre"
    path = os.path.join(output_dir, date_path)
    if not os.path.exists(path):
        os.makedirs(path)
    # Read NERRE input data
    notes = glob.glob(os.path.join(output_dir, 'data/nerre', '*.txt'))
    
    ### Get prompt design
    if few_shot == True:
        system_msg = config['NERRE']['few_prompt']
        few_user = config['NERRE']['few_user']
        few_assistant = config['NERRE']['few_assistant']
        
        logging.info(f'start API requests...')
        for note in td.tqdm(notes, desc="Generating NER-RE output from i2b2", unit="files"):
            if os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(note))[0] + ".xml")):
                logging.info("output exists: %s" % note)
            else:
                with open(note, 'r') as f:
                    content = f.read()
                    
                api_no = 1
                while api_no < api_retry:
                    try:
                        if not api_no == 1:
                            logging.info(f'{api_no}th API re-requests...')
                        completions = openai.ChatCompletion.create(
                            model = model,
                            temperature = temp,
                            n = 1,
                            messages = [
                                {'role':'system', 'content':system_msg},
                                {'role':'user', 'content':few_user},
                                {'role':'assistant', 'content':few_assistant},
                                {'role':'user', 'content':content}
                            ]
                        )
                        response = completions.choices[0]['message']['content']
                        break
                    except Exception as e:
                        logging.error(f"{note}: {api_no}th API error: \n{e}")
                        logging.info(f"susepnding 30 secs to avoid max retries...\n")
                        time.sleep(30)
                        api_no += 1
                        response = ''

                if not response == '':                                            
                    # Remove incomplete responses
                    lines = response.strip().split('\n')
                    lines = [line for line in lines if all(keyword in line for keyword in ('toID', 'fromID', 'type'))]
                    response = '\n'.join(lines)
                    response = '<TAGS>\n' + response + '\n</TAGS>'
                    
                    output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
                    with open(output_file, 'w', encoding = 'utf-8') as f:
                        f.write(response)
                else: 
                    logging.info(f"pass saving {os.path.splitext(os.path.basename(note))[0]} file due to empty response...\n")
    
    else:
        system_msg = config['NERRE']['zero_prompt']
        
        logging.info(f'start API requests...')
        for note in td.tqdm(notes, desc="Generating NERRE output from i2b2", unit = "files"):
            if os.path.exists(os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')):
                logging.info('output exists: %s' % note)
            else:
                with open(note, 'r') as f:
                    content = f.read()
                       
                api_no = 1
                while api_no < api_retry:
                    try:
                        if not api_no == 1:
                            logging.info(f'{api_no}th API re-requests...')
                        completions = openai.ChatCompletion.create(
                            model = model,
                            temperature = temp,
                            n = 1,
                            messages = [
                                {'role':'system', 'content':system_msg},
                                {'role':'user', 'content':content}
                            ]
                        )
                        response = completions.choices[0]['message']['content']
                        break
                    except Exception as e:
                        logging.error(f"{note}: {api_no}th API error: \n{e}")
                        logging.info(f"susepnding 30 secs to avoid max retries...\n")
                        time.sleep(30)
                        api_no += 1
                        response = ''
                
                if not response == '':                                    
                    lines = response.strip().split('\n')
                    lines = [line for line in lines if all(keyword in line for keyword in ('toID','fromID','type'))]
                    response = '\n'.join(lines)
                    response = '<TAGS>\n' + response + '\n</TAGS>'
                    
                    output_file = os.path.join(path, os.path.splitext(os.path.basename(note))[0] + '.xml')
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(response)
                else: 
                    logging.info(f"pass saving {os.path.splitext(os.path.basename(note))[0]} file due to empty response...\n")