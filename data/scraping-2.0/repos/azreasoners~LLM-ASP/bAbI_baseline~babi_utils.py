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
plural_to_singular = {'wolves':'wolf','cats':'cat','sheep':'sheep','mice':'mouse'}
def get_response(prompt,model, stop = None, temp=0.,max_tokens=30, sleep=None):
    ''' Takes a prompt and returns GPT-3's response '''
    response = openai.Completion.create(model=model, prompt=prompt, temperature=temp, max_tokens=max_tokens, stop=stop)
    
    if sleep:
        time.sleep(sleep)
    return response

def GPT_answer(prompt, context, query, model, config, prompt_cache):
    ''' Chooses how to call GPT-3 ''' 
    full_example = prompt+'\n'.join(context) + '\n'+query + '\n\nAnswer:'
    if full_example in prompt_cache:
        response = prompt_cache[full_example]
    else:
        response = get_response(full_example, model, stop = config.stop, max_tokens = 1000, sleep = config.sleep)
        prompt_cache[full_example]=response
    #breakpoint()
    prediction = response['choices'][0]['text'].strip()
    
    return prediction, prompt_cache

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


def get_query(idx, lines, beginning_lines, question_lines, generate_prompts = False):

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
    if not generate_prompts:
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
    clingo_control = Control(['0', '--warn=none', '--t 16'])
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

def check_answer(pred,answer,task_num):
    if pred in str_to_int:
        pred=str(str_to_int[pred])
        
    if task_num==8:
        pred = set([pp.strip() for pp in pred.split(',')])
        if pred == set(answer.split(',')):
            return True
        else:
            return False
    elif task_num==15:
        #breakpoint()
        pred = plural_to_singular[pred] if pred in plural_to_singular else pred
        if pred == answer:
            return True
        else:
            return False
    else:
        return pred.lower()==answer.lower()
        
    
def write_debug(debug, correct, total, task_idx, extra_fname):
    import os
    if not os.path.exists('debug'):
        os.mkdir('debug')
    if not os.path.exists(f'debug/{task_idx}'):
        os.mkdir(f'debug/{task_idx}')
    full_string=''#f' TOTAL ACCURACY = {correct}/{total} = {correct/total}\n\n'
    for ex in debug:
        eval_num, context, query, instance_string, ASP_program, pred, answer , correctness =  ex
        #ex_string = 'EVAL NUM:\n' + str(eval_num) +'\n\nPREDICTION:\n' + pred + '\n\nAnswer:\n' + answer + '\n\n' +'\n\nCONTEXT:\n' + '\n'.join(context) + '\n\nQUERY:\n' + query + '\n\nINSTANCE STRING PARSED FROM GPT-3\n:' + instance_string + '\n\nASP_PROGRAM:\n' + ASP_program +   '\n\n'
        ex_string = 'EVAL NUM: ' + str(eval_num) + ' ' +  correctness +  '\n\nCONTEXT:\n' + '\n'.join(context) + '\n\nQUERY:\n' + query + '\n\nPREDICTION:\n' + pred + '\n\nAnswer:\n' + answer + '\n\n'
        full_string+=ex_string
    with open(f'debug/{task_idx}/debug_{task_idx}_{extra_fname}.txt', 'w') as f:
        f.write(full_string)
    return

def write_debug_baseline(debug, correct, total, task_idx, base_folder, extra_fname):
    import os
    if not os.path.exists('debug'):
        os.mkdir('debug')
    if not os.path.exists(f'debug/{base_folder}'):
        os.mkdir(f'debug/{base_folder}')
    full_string=''#f' TOTAL ACCURACY = {correct}/{total} = {correct/total}\n\n'
    for ex in debug:
        eval_num, context, query, pred, answer, correctness  =  ex
        #ex_string = 'EVAL NUM:\n' + str(eval_num) +'\n\nPREDICTION:\n' + pred + '\n\nAnswer:\n' + answer + '\n\n' +'\n\nCONTEXT:\n' + '\n'.join(context) + '\n\nQUERY:\n' + query + '\n\nINSTANCE STRING PARSED FROM GPT-3\n:' + instance_string + '\n\nASP_PROGRAM:\n' + ASP_program +   '\n\n'
        ex_string = 'EVAL NUM: ' + str(eval_num) + ' ' +  correctness +  '\n\nCONTEXT:\n' + '\n'.join(context) + '\n\nQUERY:\n' + query + '\n\nPREDICTION:\n' + pred + '\n\nAnswer:\n' + answer + '\n\n'
        full_string+=ex_string
    with open(f'debug/{base_folder}/debug_{extra_fname}_task{task_idx}.txt', 'w') as f:
        f.write(full_string)
    return

def write_debug_COT(debug, correct, total, task_idx, base_folder, extra_fname):
    import os
    if not os.path.exists('debug'):
        os.mkdir('debug')
    if not os.path.exists(f'debug/{base_folder}'):
        os.mkdir(f'debug/{base_folder}')
    full_string=''#f' TOTAL ACCURACY = {correct}/{total} = {correct/total}\n\n'
    for ex in debug:
        eval_num, context, query, instance_string, ASP_program, pred_full, pred, answer , correctness =  ex
        #ex_string = 'EVAL NUM:\n' + str(eval_num) +'\n\nPREDICTION:\n' + pred + '\n\nAnswer:\n' + answer + '\n\n' +'\n\nCONTEXT:\n' + '\n'.join(context) + '\n\nQUERY:\n' + query + '\n\nINSTANCE STRING PARSED FROM GPT-3\n:' + instance_string + '\n\nASP_PROGRAM:\n' + ASP_program +   '\n\n'
        ex_string = 'EVAL NUM: ' + str(eval_num) + ' ' +  correctness +  '\n\nCONTEXT:\n' + '\n'.join(context) + '\n\nQUERY:\n' + query + '\n\nEXTRACTED PREDICTION:\n' + pred_full + '\n\nPREDICTION:\n' + pred + '\n\nAnswer:\n' + answer + '\n\n'
        full_string+=ex_string
    with open(f'debug/{base_folder}/debug_{extra_fname}_task{task_idx}.txt', 'w') as f:
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