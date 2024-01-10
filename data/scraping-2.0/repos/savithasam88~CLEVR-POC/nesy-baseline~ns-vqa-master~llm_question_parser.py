def tokenize_program(s, delim=' ',
            add_start_token=True, add_end_token=True,
            punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    s= str(s)    
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s%s' % (delim, p, delim))
            
    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    tokens_revised = []
    for t in range(len(tokens)):
      if tokens[t] in ['hasProperty', 'same_material', 'same_size', 'same_color', 'same_shape', 'front', 'behind', 'right', 'left']:
        tok = tokens[t]
        t = t+1
        while(tokens[t] != ')'):
          tok = tok+tokens[t]
          t=t+1
        tok = tok+tokens[t]
        tokens_revised.append(tok)
      else:
        if tokens[t] == '!=':
          start = t-1
          while(tokens[start]!=','):
            start=start-1
          end = t+1
          while(end!=len(tokens) and tokens[end]!=',' and tokens[end]!='.'):
            end=end+1
          tokens_revised.append(''.join(tokens[start+1:end]).strip())
    return tokens_revised


def parse(prediction):
  body = prediction.split(':-')[1]
  body.split(',')
  
  
  
import os

data_path = '/home/savitha/code/CLEVR-POC/nesy-baseline/ns-vqa-master/data/reason/' 
dataset_name = 'output-2000'

with open(os.path.join(data_path, 'prompt.txt'), 'r') as fp:
  examples = fp.read().splitlines()

prompt = ' '.join(examples)
prompt = 'Convert the following questions to the ASP logic language:' + prompt


question_file_names = 'llm_questions_' + dataset_name + '.txt'
program_file_names = 'llm_programs_' + dataset_name + '.txt'

with open(os.path.join(data_path, dataset_name, question_file_names), 'r') as fp:
  #questions = fp.readlines()
  questions = fp.read().splitlines()

with open(os.path.join(data_path, dataset_name, program_file_names), 'r') as fp:
  programs = fp.read().splitlines()

import openai
openai.api_key = "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"

  

spliters = [';', ',', '.', '(', ')',':-', '!=' ]
current_correct_predictions = 0
prompt_questions = []
predictions = []
for idx, q in enumerate(questions):
  prompt_question = prompt + "Question:" + q + " ASP:"
  response_q = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt_question,
    temperature=0,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  prediction = response_q['choices'][0]['text']
  predicted_tokens = set(tokenize_program(prediction, punct_to_keep=spliters))
  actual_tokens = set(tokenize_program(programs[idx], punct_to_keep=spliters))
  if (predicted_tokens == actual_tokens):
    current_correct_predictions = current_correct_predictions + 1
  current_accuracy = current_correct_predictions/(idx + 1)
  print(f'Accuracy at step {idx+1}: {current_accuracy}')
  print('--------------------------------------')

