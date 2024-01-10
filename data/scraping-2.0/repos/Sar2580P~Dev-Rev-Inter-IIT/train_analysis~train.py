import pandas as pd
import sys, os
sys.path.append(os.getcwd())
from agent_executor.agent_executer import agent_executor
from langchain.docstore.document import Document
from icecream import ic
from langchain.callbacks import get_openai_callback
from utils.chains import *
from memory.memory import mistake_memory


data  = pd.read_csv('DATA_DEVREV_72_tt_splitted/reducd_combined_train.csv' ).iloc[:1,:]

# print(data)

# query = 'List all high severity tickets coming in from slack from customer abc123 and generate a summary of them.'
# ground_json = '''
# [
#     {
#         "tool_name": "search_object_by_name",
#         "arguments": [
#             {
#                 "argument_name": "query",
#                 "argument_value": "Cust123"
#             }
#         ]

#     },

#     {
#         "tool_name": "works_list",
#         "arguments": [
#             {
#                 "argument_name": "ticket.rev_org",
#                 "argument_value": "$$PREV[0]"
#             },
#             {
#                 "argument_name": "ticket.severity",
#                 "argument_value": ["high"]
#             },
#             {
#                 "argument_name": "ticket.source_channel",
#                 "argument_value": ["slack"]
#             },
#             {
#                 "argument_name": "type",
#                 "argument_value": ["ticket"]
#             }
#         ]
#     },
#     {
#         "tool_name": "summarize_objects",
#         "arguments": [
#             {
#                 "argument_name": "objects",
#                 "argument_value": "$$PREV[1]"
#             }
#         ]
#     }
# ]
# '''
# data = [(query , ground_json)]

# print(data.shape)

#___________________________________________________________________________________________

agent_executor.train()

def build_experience(x):
    thoughts_action = '\n'.join(x['intermediate_thoughts'])
    y = create_tool_experience_chain.run({"query": x['query'] , "agent_scratchpad" : thoughts_action , 
                "correct_tool_name": x['correct_tool'] ,"tool_description": x['correct_tool_description'],
                })
    # print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
    # print(create_tool_experience_chain.prompt)

    return y.strip('\n').strip()

#___________________________________________________________________________________________
ct = 0

for i in range(len(data)):
  print("\033[1;32m {}\033[00m" .format('QUERY COUNT : {i}'.format(i=i)))
  query, ground_json = data.iloc[i,0], data.iloc[i,1]
#   query, ground_json = data[0][0] , data[0][1]
  print("\033[1;32m {}\033[00m" .format('QUERY : ') , "\033[93m {}\033[00m" .format(query))
  print("\033[1;32m {}\033[00m" .format('Ground JSON :') , "\033[93m {}\033[00m" .format(ground_json))

  agent_executor.get_tool_lists(ground_json)
  
  with get_openai_callback() as cb:
    x = agent_executor(inputs={"input": query})
    thought_execution_chain , checkpoints = agent_executor.thought_execution_chain , agent_executor.checkpoints

    for tool_index, value in checkpoints.items():
      x = {
        "query":query, 
        "correct_tool" :value['correct_tool'] , 
        "wrong_tool" : value['wrong_tool'] ,
        "wrong_tool_description" : value['wrong_tool_description'] ,
        "correct_tool_description" :value['correct_tool_description'] , 
        "intermediate_thoughts" : thought_execution_chain[:tool_index],
        "tool_thought": value['thought']
      }


      human_eval = 'n'
      # human_eval = input("Do you want to correct the reasoning? (y/n) :")
      if human_eval.lower() == 'n':
        experience = build_experience(x)
      else :
        experience = input("This has been the mistake summary : \n\t{x}. \nPlease write the correct reasoning :".format(x=x))
      
    #   learning  = '- MISTAKE_HIGHLIGHT : {b}\n'.format(b = experience)
      learning  = '- MISTAKE_HIGHLIGHT : {b}\n'.format(b = experience)
      metadata = {
        # 'query': x['query'],
        'correct_tool': x['correct_tool'],
        'wrong_tool': x['wrong_tool'],
        'learning': learning
      }
      print('metadata : ' , metadata)
      print('tool_thought : ' , x['tool_thought'])
      doc = Document(page_content=x['tool_thought'] , metadata=metadata)
      mistake_memory.stage(doc)
    
    print("\033[96m {}\033[00m" .format(agent_executor.return_schema))
  user_decision = 'y'
  # user_decision = input('Do you want to save the experience? (y/n) : ')  
  if user_decision.lower() == 'y':
    ct+= mistake_memory.queue.qsize()
    mistake_memory.push()

  else:
    mistake_memory.clear()
    print("\033[91m {}\033[00m" .format('skipping experience saving...'))

  print("\033[91m {}\033[00m" .format('---------------- QUERY_COST : $ {cost}---------------- MISTAKES LEARNED : {ct}-------------------- QUERY TOKENS : {tokens}-----------------'.format(cost = round(cb.total_cost, 5) , 
                                                                                                                                                                                            ct = ct, tokens = cb.total_tokens)))
  
