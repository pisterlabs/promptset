#%%
#%%
import openai
from methods.methods import *
import time
from tqdm import tqdm
import copy
import pandas as pd
### LOAD EVALUATION AND QRELS
evaluation_path = './trec/treccast/'
qrels_path = './trec/qrels/'
all_qrels = load_all_qrels(qrels_path).reset_index(drop=True)
qrels19 = all_qrels[all_qrels.year == 2019]
eval_19 = pd.read_csv('./trec/evaluation2019.csv')
eval_20 = pd.read_csv('./trec/evaluation2020.csv')
eval_21 = pd.read_csv('./trec/evaluation2021.csv')

openai.api_key = 'Your key'

import re
def sub_(x):
    x = re.sub('\n- .*','',x)
    x = re.sub('De-contextualized rewrite under the multi-turn information-seeking dialog context:','',x)
    x = re.sub('Response:.*','',x)
    x = re.sub('\nCurrent question.*\n.*','',x)
    x = re.sub('Previous question:.*\nRewritten.*','',x)
    x = re.sub('\t.*sorry.*','',x)    
    x = re.sub('"Earlier.*\.','',x)    
    x = re.sub('Earlier, we.*\.','',x)    
    x = re.sub('Keywords added:.*','',x)
    x = re.sub('"keywords:.*','',x)
    x = re.sub('Response:.*','',x)
    x = re.sub('User: .*','',x)
    x = re.sub('AI assistant:.*','',x)
    x = re.sub('Response: .*','',x)
    x = re.sub('"Current question:.*','',x)
    x = re.sub('Current question:.*','',x)
    x = re.sub('"Context: ','',x)
    x = re.sub('Context: ','',x)
    x = re.sub('','',x)
    x = re.sub('','',x)
    x = re.sub('"Reformulated question:','',x)
    x = re.sub('Reformulated question:','',x)
    x = re.sub('Reformulated question : ','',x)
    x = re.sub('Reformulated question: ','',x)
    x = re.sub('"Rephrased question: ','',x)
    x = re.sub('Rephrased question: ','',x)
    x = re.sub('"Request for conversational system: .*\n\nRewritten request: "','',x)
    x = re.sub('Request for conversational system:.*\n\nRewritten request: "','',x)
    x = re.sub('Previous keywords: ','',x)
    x = re.sub('"From the previous question:.*','',x)
    x = re.sub('From the previous question:.*','',x)
    x = re.sub('"Previous context:.*','',x)
    x = re.sub('Previous context:.*','',x)
    x = re.sub('"Keywords: ','',x)
    x = re.sub('Keywords: ','',x)
    x = re.sub('\n\n','',x)
    x = re.sub('"Search keywords: ','',x)
    x = re.sub('Search keywords: ','',x)
    x = re.sub('Prompt: ','',x)
    x = re.sub('Prompt for search engine: ','',x)
    x = re.sub('Search Engine Prompt: ','',x)
    x = re.sub("I'm sorry, but your current question",'',x)
    x = re.sub('lacks sufficient context .*','',x)
    x = re.sub('Query for a search engine: ','',x)
    x = re.sub('Search prompt: ','',x)
    x = re.sub('Search engine prompt: ','',x)
    x = re.sub('Search engine prompt:','',x)
    x = re.sub('Rewritten question: ','',x)
    x = re.sub('Rewritten question:','',x)
    x = re.sub('Request for a retrieval system: ','',x)
    x = re.sub('Request for a retrieval system:','',x)
    x = re.sub('"Request for clarification: ','',x)
    x = re.sub('Request for clarification:','',x)
    x = re.sub('Request: ','',x)
    x = re.sub('"Request for clarification: ','',x)
    x = re.sub('.*answer: ','',x)
    x = re.sub('Answer:.*','',x)
    x = re.sub('Question: ','',x)
    x = re.sub('"OP:.*','',x)
    x = re.sub('Request for retrieval system: ','',x)
    x = re.sub('Revised question: ','',x)
    x = re.sub('Could you please clarify your question?','',x)
    x = re.sub('Building on the previous questions:','',x)
    x = re.sub('Building on the previous questions','',x)
    x = re.sub('Building on previously asked questions ','',x)
    x = re.sub('Reformulated question in a multi-turn information-seeking dialog context: ','',x)
    x = re.sub('Rewritten: ','',x)
    x = re.sub('Reformulated: ','',x)
    x = re.sub("Revised: ",'',x)

    x = re.sub('\(.*\)','',x)
    x = re.sub('"','',x)
    x = re.sub('"\n','',x)
    x = re.sub('\?.*','?',x)
    x = re.sub("I'm sorry, but .*",'',x)
    return x
  
  
def sustitube(x):
    x = re.sub("\(.*\)","",x)
    return x
    
      
def chatgpt(messages:list, model = 'gpt-3.5-turbo'):
  response = openai.ChatCompletion.create(
  model=model,
  messages=messages)
  return response.choices[0]['message']['content']



# %%
'''
system_text = 'In a multi-turn dialog system, rewrite the given sentence to be self-explanatory. Use elements of the previous sentences to generate better sentences.'
messages= [{"role": "system", "content": system_text}]
example = []
conv = _2020[_2020.conv.isin(['81','82'])]

for convid in conv.conv.unique():
    part = conv[conv.conv == convid]
    for turn in part.turn:
        if int(turn)<7:
            if convid =='82' and turn =='1':
                example.append({"role": "user", "content": 'New conversation.'})
            example.append({"role": "user", "content": part[(part.turn == turn)].raw_utterance.iloc[0]})
            example.append({"role": "assistant", "content": part[(part.turn == turn)].manual_rewritten_utterance.iloc[0]})
        
messages += example   
'''

def create_messages(system_text,command,current_utterance,previous_utterances=[],previous_outputs=[],previous_in_current = False,prompt_in_input=False):
    if prompt_in_input:
      messages = []
    else:
      messages = [{"role": "system", "content": system_text}]
    if previous_in_current:
      if previous_utterances == []:
        new_current = ''
      else:
        new_current = 'Previous context:'
      for x,y in zip(previous_utterances,previous_outputs):
        messages.append({"role": "user", "content": x})
        messages.append({"role": "assistant", "content":y})
        new_current += f"{x} "#f"original: {x}, rewritten:{y} "
      new_current += command + current_utterance
      if prompt_in_input:
        new_current = system_text + new_current
      messages.append({"role": "user", "content": new_current})
    else:    
      for x,y in zip(previous_utterances,previous_outputs):
        messages.append({"role": "user", "content":command + x})
        messages.append({"role": "assistant", "content":y})
      if prompt_in_input:
        messages.append({"role": "user", "content":system_text+ " " + command + current_utterance})
      else:
        messages.append({"role": "user", "content":system_text+ " " + command + current_utterance})
    return messages


def chatgpt_for_df(evaluation,system_text = 'In a multi-turn dialog system, rewrite the given sentence to be self-explanatory following the pattern of the previous interactions.',year=2019):
  current_utterance = evaluation.raw_utterance.iloc[0]
  
  starting_message= [{"role": "system", "content": system_text}]
  dictionary = {'qid':{},'query' :{},'current_input':{}}
  ind = 0
  rate_limit_per_minute = 20
  delay = 60.0 / rate_limit_per_minute
  example = []
  if year == 2019:
    conv = _2020[_2020.conv_id.isin(['81','82'])]
    conv_id_stop ='82'    
  elif year ==2020:
    conv = _2019[_2019.conv_id.isin([31,32])]
    conv_id_stop =32    

  for convid in conv.conv_id.unique():
    part = conv[conv.conv_id == convid]
    for turn in part.turn:
        if int(turn)< 9:
            if convid ==conv_id_stop and turn =='1':
                example.append({"role": "user", "content": 'New conversation.'})
            example.append({"role": "user", "content": part[(part.turn == turn)].raw_utterance.iloc[0]})
            example.append({"role": "assistant", "content": part[(part.turn == turn)].manual_rewritten_utterance.iloc[0]})
  starting_message += example   
  
  prompts = {'qid':{},'prompt' :{}}
  #previous_utterances = []
  #previous_outputs = []
  for conv_id in tqdm(evaluation.conv_id.unique()):
      conv = evaluation[evaluation.conv_id ==conv_id]
      previous_utterances = []
      previous_outputs = []
      for qid in (conv.qid):
        time.sleep(delay)
        try :
          messages = copy.copy(starting_message)
          current_utterance = conv[conv.qid == qid].raw_utterance.iloc[0]
          for x,y in zip(previous_utterances,previous_outputs):
            messages.append({"role": "user", "content": x})
            messages.append({"role": "assistant", "content":sustitube(y)})
          #message = create_messages(system_text,current_command,current_utterance,previous_utterances,previous_outputs,previous_in_current = previous_in_current,prompt_in_input=prompt_in_input)
          message = copy.copy(messages)
          message += [{"role": "user", "content": system_text + current_utterance}]
          prompts['qid'][ind] = qid
          prompts['prompt'][ind] = message
          #print(message)
          response = chatgpt(message)
          previous_utterances.append(current_utterance)
          previous_outputs.append(response)
          dictionary['qid'][ind] = qid
          dictionary['query'][ind] = response
          dictionary['current_input'][ind] = message[-1]['content']
          ind+=1
        except:
          try :
            time.sleep(delay*2)
            messages = copy.copy(starting_message)
            current_utterance = conv[conv.qid == qid].raw_utterance.iloc[0]
            for x,y in zip(previous_utterances,previous_outputs):
                messages.append({"role": "user", "content": x})
                messages.append({"role": "assistant", "content":sustitube(y)})
            #message = create_messages(system_text,current_command,current_utterance,previous_utterances,previous_outputs,previous_in_current = previous_in_current,prompt_in_input=prompt_in_input)
            message = copy.copy(messages)
            message += [{"role": "user", "content": system_text + current_utterance}]
            prompts['qid'][ind] = qid
            prompts['prompt'][ind] = message
            #print(message)
            response = chatgpt(message) 
            previous_utterances.append(current_utterance)
            previous_outputs.append(response)
            dictionary['qid'][ind] = qid
            dictionary['query'][ind] = response
            dictionary['current_input'][ind] = message[-1]['content']
            ind+=1
          except Exception as e:
            try :
                messages = copy.copy(starting_message)
                time.sleep(delay*3)
                current_utterance = conv[conv.qid == qid].raw_utterance.iloc[0]
                for x,y in zip(previous_utterances,previous_outputs):
                    messages.append({"role": "user", "content": x})
                    messages.append({"role": "assistant", "content":sustitube(y)})
                #message = create_messages(system_text,current_command,current_utterance,previous_utterances,previous_outputs,previous_in_current = previous_in_current,prompt_in_input=prompt_in_input)
                message = copy.copy(messages)
                message += [{"role": "user", "content": system_text + current_utterance}]
                prompts['qid'][ind] = qid
                prompts['prompt'][ind] = message
                #print(message)
                response = chatgpt(message) 
                previous_utterances.append(current_utterance)
                previous_outputs.append(response)
                dictionary['qid'][ind] = qid
                dictionary['query'][ind] = response
                dictionary['current_input'][ind] = message[-1]['content']
                ind+=1
            except Exception as e:
              print(e)
              try :
                messages = copy.copy(starting_message)
                time.sleep(delay*4)
                current_utterance = conv[conv.qid == qid].raw_utterance.iloc[0]
                for x,y in zip(previous_utterances,previous_outputs):
                    messages.append({"role": "user", "content": x})
                    messages.append({"role": "assistant", "content":sustitube(y)})
                #message = create_messages(system_text,current_command,current_utterance,previous_utterances,previous_outputs,previous_in_current = previous_in_current,prompt_in_input=prompt_in_input)
                message = copy.copy(messages)
                message += [{"role": "user", "content": system_text + current_utterance}]
                prompts['qid'][ind] = qid
                prompts['prompt'][ind] = message
                #print(message)
                response = chatgpt(message) 
                previous_utterances.append(current_utterance)
                previous_outputs.append(response)
                dictionary['qid'][ind] = qid
                dictionary['query'][ind] = response
                dictionary['current_input'][ind] = message[-1]['content']
                ind+=1
              except Exception as e:
                print(e)
                print('end of cilcle, missing qid : ',qid)
      messages.append({"role": "user", "content": 'New conversation.'})

  return pd.DataFrame(dictionary), pd.DataFrame(prompts)
 
# %%

_2020 = create_df_from_json(pd.read_json('./trec/treccast/2020_manual_evaluation_topics_v1.0.json'))
_2020['conv_id'] = [x.split('_')[0] for x in _2020.number]
_2020['turn'] = [x.split('_')[1] for x in _2020.number]
qrels20 =load_qrels('./trec/qrels/2020qrels.txt')

_2019 = pd.read_csv('./trec/evaluation2019.csv')
manual_2019 = pd.read_csv('./trec/treccast/test_manual_utterance.tsv', sep = '\t', names=['qid','manual_rewritten_utterance'])
_2019 = _2019.merge(manual_2019,on='qid')

prompts = pd.read_csv('./top5_prompts.csv',sep=';')
for i in range(len(prompts)):
      
      name = prompts['id'].iloc[i]
      text = prompts['text'].iloc[i]
      if not(os.path.isfile(f'/data4/guidorocchietti/GPT_clean/ultima_prova/rewritings/rewritten/{name}_2019.tsv')):
        print('Name : ',name)
        print('Text : ',text)

        output, prompt = chatgpt_for_df(eval_19,system_text=text,year=2019)
        output[['qid','query']].to_csv(f'/data4/guidorocchietti/GPT_clean/ultima_prova/rewritings/rewritten/{name}_2019.tsv', sep = '\t', index=False)
        prompt.to_csv(f'/data4/guidorocchietti/GPT_clean/ultima_prova/rewritings/prompts/{name}_2019.csv')
      #output, prompt = chatgpt_for_df(eval_20,system_text=text,year=2020)
      #output[['qid','query']].to_csv(f'/data4/guidorocchietti/GPT_clean/ultima_prova/rewritings/rewritten/{name}_2020.tsv', sep = '\t')
      #prompt.to_csv(f'/data4/guidorocchietti/GPT_clean/ultima_prova/rewritings/prompts/{name}_2020.csv')
#%%
#output, prompt = chatgpt_for_df(eval_20)
#output[['qid','query']].to_csv('/data4/guidorocchietti/GPT_clean/ultima_prova/rewritings/rewritten/Example_in_history_2020.tsv', sep = '\t', index=False)
#prompt.to_csv('/data4/guidorocchietti/GPT_clean/ultima_prova/rewritings/prompts/Example_in_history_2020.csv')
# %%
