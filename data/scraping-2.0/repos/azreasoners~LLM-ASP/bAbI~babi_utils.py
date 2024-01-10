import time
import openai
import sys
sys.path.append('../')

from clingo.control import Control

from keys import API_KEY,ORG_KEY

openai.organization = ORG_KEY
openai.api_key = API_KEY

# Define any helper variables
NAMES = ['Mary', 'Daniel', 'Sandra', 'John','Bill', 'Gertrude','Bernhard', 'Brian', 'Lily', 'Julius', 'Greg', 'Jason', 'Sumit', 'Antoine', 'Yann', 'Fred', 'Jeff', 'Winona', 'Jessica', 'Emily', 'Julie']
str_to_int  = {'none': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}

def get_response(prompt,model, stop = None, temp=0.,max_tokens=30, sleep=None):
    ''' Takes a prompt and returns GPT-3's response '''
    response = openai.Completion.create(model=model, prompt=prompt, temperature=temp, max_tokens=max_tokens, stop=stop)
    
    if sleep:
        time.sleep(sleep)
    return response

def GPT_parse_iterative(prompts, context, context_preprocess, query, model, stops, parse_word, sleep, temporal_stamp, separate, prompt_cache):
    ''' Process context sentences by iteratively calling GPT-3 line by line '''
    stop_context = stops[0]
    stop_query = stops[1]
    context_GPT_responses=list()
    parsed_responses=list()
    context_prompt, query_prompt=prompts
    context_parse_word = parse_word[0]
    for context_line in context:
        if context_line in prompt_cache:
            response = prompt_cache[context_line]
        else:
            response = get_response(context_prompt+context_line+context_parse_word, model, stop =stop_context,  sleep=sleep)
            prompt_cache[context_line]=response
        context_GPT_responses.append(response)
        parsed_responses.append(response['choices'][0]['text'])
    facts = [fact.strip() for fact in parsed_responses]
    facts = [lowercase_terms(fact) for fact in facts]
    
    if separate==True:
        query_parse_word = parse_word[1]
        if query in prompt_cache:
            query_response = prompt_cache[query]
        else:
            query_response = get_response(query_prompt + query + query_parse_word, model, stop = stop_query, sleep = sleep)
            prompt_cache[query]=query_response
        
        query_text = query_response['choices'][0]['text']
        query_text= query_text.strip()
        query_text = lowercase_terms(query_text)
    else:
        query_text = ''
    instance_string = create_extra_rules(facts,query_text, temporal_stamp, separate = separate)
    return instance_string, prompt_cache


def GPT_parse_all(prompts, context, context_preprocess, query, model, stop, parse_word, sleep, temporal_stamp, separate, prompt_cache):
    ''' Process context all in one GPT-3 call '''
    prompt=prompts[0]
    context_parse_word = parse_word[0]
    context_query = '\n'.join(context+([query] if not separate else []))
    context_query = context_preprocess(context_query) if context_preprocess else context_query
    full_example = context_query + context_parse_word
    context_GPT_responses=list()
    parsed_responses=list()
    
    if full_example in prompt_cache:
        response = prompt_cache[full_example]
    else:
        response = get_response(prompt+full_example,model, max_tokens=150)
        prompt_cache[full_example]=response
    context_GPT_responses.append(response)
    parsed_responses.append(response['choices'][0]['text'])

    facts = [fact.strip() for fact in parsed_responses]
    facts = facts[0].split('.')
    facts = [fact for fact in facts if fact!='']
    facts = [fact+'.' for fact in facts]
    facts = [lowercase_terms(fact) for fact in facts]
    
    if separate:
        query_parse_word= parse_word[1]
        query_prompt = prompts[1]
        if query in prompt_cache:
            query_response = prompt_cache[query]
        else:
            query_response = get_response(query_prompt + query + query_parse_word, model, stop = stop, sleep = sleep)
            prompt_cache[query]=query_response
        
        query_text = query_response['choices'][0]['text']
        query_text= query_text.strip()
        query_text = lowercase_terms(query_text)
        facts += [query_text]
    if not temporal_stamp:
        instance_string = '\n'.join(facts)
    else:
        instance_string = create_extra_rules(facts[:-1],facts[-1], temporal_stamp, separate = True)
    return instance_string, prompt_cache

def GPT_parse(prompts, context, query, model, config, prompt_cache):
    ''' Chooses how to call GPT-3 ''' 
    if config.custom_parse:
        instance_string, prompt_cache = config.custom_parse(prompts, context, query, model, config.stops, config.parse_word, config.sleep, prompt_cache)
        return instance_string, prompt_cache
    if config.iterative:
        instance_string, prompt_cache =  GPT_parse_iterative(prompts, context, config.context_preprocess, query, model, config.stops, config.parse_word, config.sleep, config.temporal_stamp, config.separate, prompt_cache)
    else:
        instance_string, prompt_cache =  GPT_parse_all(prompts, context, config.context_preprocess, query, model, config.stops[0], config.parse_word, config.sleep, config.temporal_stamp, config.separate, prompt_cache)
        
    return instance_string, prompt_cache

def custom17( prompts, context, query,model, stops, parse_word, sleep, prompt_cache):
    ''' Custom GPT-3 call for task #17 '''
    context_stop,query_stop = stops
    context_prompt = prompts[0]
    context_GPT_responses=list()
    parsed_responses=list()
    
    full_context_list=[]
    for idx,context_line in enumerate(context):
        full_context_list.append(f'Sentence {str(idx+1)}: {context_line}')
    full_context ='\n'.join(full_context_list)
    
    if full_context in prompt_cache:
        response = prompt_cache[full_context]
    else:
        response = get_response(context_prompt+full_context + parse_word[0], model, stop = context_stop, max_tokens = 100, sleep = sleep)
        prompt_cache[full_context]=response
    context_GPT_responses.append(response)
    parsed_responses.append(response['choices'][0]['text'])

    sentences = parsed_responses[0].split('\n')
    sentences = [sentence for sentence in sentences if sentence!='']
    objects_string,direction_string = sentences
    objects_string=objects_string.strip()
    input_directional_facts = direction_string[17:]
    
    query_GPT = objects_string + '\n' + 'Question: '+ query
    
    query_prompt = prompts[1]
    
    if query_GPT in prompt_cache:
        query_response = prompt_cache[query_GPT]
    else:
        query_response = get_response(query_prompt +parse_word[0]+query_GPT + parse_word[1], model, stop = query_stop, max_tokens = 60, sleep = sleep)
        prompt_cache[query_GPT]=query_response

    query_text = query_response['choices'][0]['text']
    query_text= query_text.strip()
    
    query_fact = query_response['choices'][0]['text'].strip()
    
    return input_directional_facts + query_fact, prompt_cache


def lowercase_terms(string):
    ''' lowercase names so they are not treated as variables in ASP '''
    for name in NAMES:
        if name in string:
            string= string.replace(name,name.lower())
    return string

def get_query_indices(idx, beginning_lines, question_lines):

    query_idx = question_lines[idx]
    candidate_idx=0
    beginning_idx = candidate_idx

    for beg_idx in beginning_lines+[beginning_lines[-1]+500]:
        if beg_idx > query_idx:
            beginning_idx = candidate_idx
            break
        candidate_idx=beg_idx

    assert beginning_idx!=-1

    return beginning_idx, query_idx


def get_query(idx, lines, beginning_lines, question_lines):

    beg_idx, query_idx = get_query_indices(idx, beginning_lines, question_lines)


    context_lines = lines[beg_idx:query_idx]
    context_lines= [line.strip() for line in context_lines]
    context_lines = [''.join([char for char in line if not char.isdigit() ]) for line in context_lines]
    context_lines = [line[1:] for line in context_lines]

    query_lines = lines[query_idx]
    query_line = ''.join([char for char in query_lines if not char.isdigit()])
    query_line = query_line.strip()

    query,ans = query_line.split('\t')

    query=query.strip()

    context_lines= [context_line for context_line in context_lines if '?' not in context_line]
    
    if ans in str_to_int:
        ans = str(str_to_int[ans])
    return context_lines,query,ans



def create_extra_rules(facts, query_text, temporal_stamp, separate=False):
    ''' Facts are changed to include timestamps and concatenated with query fact'''
    if separate:
        if temporal_stamp:
            for idx, fact in enumerate(facts):
                fact = fact.strip('.')
                facts[idx] = f'{fact[:-1]}, {idx}).'
        return '\n'.join(facts+[query_text.strip('.') +'.'])
    else:
        return '\n'.join(facts) 


def ASP_solve(program):
    ''' Use Clingo to solve ASP program '''
    clingo_control = Control(['0', '--warn=none'])
    models = []
    try:
        clingo_control.add('base', [], program)
    except:
        breakpoint()
    clingo_control.ground([('base', [])])
    clingo_control.solve(on_model = lambda model: models.append(model.symbols(atoms=True)))
    models = [[str(atom) for atom in model] for model in models]
    return models[-1]


def process_answer_set(answer_set, relevant_fact, post_processing, answer):
    ''' Process the answer set to extract answer '''
    predicate_name = relevant_fact['predicate_name']
    atoms = [atom for atom in answer_set if atom.startswith(predicate_name)]
    preds = []
    for fact in atoms:
        terms_string = fact.replace('(',',').replace(')',',').split(',')
        ordered_terms = [term for term in terms_string if term!='']
        
        for idx,term in relevant_fact['specified_terms'].items():
            if ordered_terms[idx+1]==term:
                continue
            else:
                break
        
        pred = [ordered_terms[i+1] for i in relevant_fact['relevant_terms_inds']]
        
        preds.append(pred)
    pred = post_processing(preds, answer)
    
    return pred


def write_debug(debug, correct, total, task_idx, extra_fname):
    import os
    if not os.path.exists('debug'):
        os.mkdir('debug')
    if not os.path.exists(f'debug/{task_idx}'):
        os.mkdir(f'debug/{task_idx}')
    full_string=f' TOTAL ACCURACY = {correct}/{total} = {correct/total}\n\n'
    for ex in debug:
        eval_num, context, query, instance_string, ASP_program, pred, answer =  ex
        ex_string = 'EVAL NUM:\n' + str(eval_num) +'\n\nPREDICTION:\n' + pred + '\n\nAnswer:\n' + answer + '\n\n' +'\n\nCONTEXT:\n' + '\n'.join(context) + '\n\nQUERY:\n' + query + '\n\nINSTANCE STRING PARSED FROM GPT-3\n:' + instance_string + '\n\nASP_PROGRAM:\n' + ASP_program +   '\n\n'
        full_string+=ex_string
    with open(f'debug/{task_idx}/debug_{task_idx}_{extra_fname}.txt', 'w') as f:
        f.write(full_string)
    return

def write_debug_CSV(debug, correct, total, task_idx, extra_fname):
    
    import csv
    import os
    if not os.path.exists('debug'):
        os.mkdir('debug')
    if not os.path.exists(f'debug/{task_idx}'):
        os.mkdir(f'debug/{task_idx}')
    full_string=f' TOTAL ACCURACY = {correct}/{total} = {correct/total}\n\n'
    fields = ['Example #', 'Prediction', 'Ground Truth', 'Context', 'query', 'Instance facts, extracted from GPT-3 and processed', 'ASP program']
    example_lists= []
    for ex in debug:
        eval_num, context, query, instance_string, ASP_program, pred, answer =  ex
        #ex_string = 'EVAL NUM:\n' + str(eval_num) +'\n\nPREDICTION:\n' + pred + '\n\nAnswer:\n' + answer + '\n\n' +'\n\nCONTEXT:\n' + '\n'.join(context) + '\n\nQUERY:\n' + query + '\n\nINSTANCE STRING PARSED FROM GPT-3\n:' + instance_string + '\n\nASP_PROGRAM:\n' + ASP_program +   '\n\n'
        example_lists.append([str(eval_num), pred, answer, context, query, instance_string, ASP_program])
    with open(f'debug/{task_idx}/debug_{task_idx}_{extra_fname}.csv', 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(example_lists)
        return