import pandas as pd
import sys, os
sys.path.append(os.getcwd())
from agent_executor.agent_executer import agent_executor
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback
import time 
import json

data  = pd.read_csv('DATA_DEVREV_72_tt_splitted/test_DATA_multi_class_3.csv').iloc[:10,: ]

path = 'prediction_data/test_DATA_multi_class_3.csv'
if not os.path.exists(path):
  prediction_df = pd.DataFrame(columns=['query' , 'groundJson' , 'predicted_json' , 'latency (in seconds)' , 'queryCost' , 'queryTokens'])
  prediction_df.to_csv(path, index=False)

prediction_df = pd.read_csv(path)
agent_executor.eval()

ct = 0
for i in range(len(data)):
  print("\033[1;32m {}\033[00m" .format('QUERY COUNT : {i}'.format(i=i)))
  query, ground_json = data.iloc[i , 0] , data.iloc[i , 1]
  print("\033[1;32m {}\033[00m" .format('QUERY : ') , "\033[93m {}\033[00m" .format(query))
  print("\033[1;32m {}\033[00m" .format('Ground JSON :') , "\033[93m {}\033[00m" .format(ground_json))

  
  with get_openai_callback() as cb:
    start = time.time()
    x = agent_executor(inputs={"input": query})
    try: 
      predict_json = json.dumps(agent_executor.return_schema)
      print(type(predict_json))

      latency = time.time() - start 
      query_cost = cb.total_cost
      query_tokens = cb.total_tokens
      prediction_df.loc[len(prediction_df)] = [query , ground_json , predict_json , round(latency,2) , round(query_cost,5) , query_tokens]
    except:
      print("Skipping query ....")
  print("\033[91m {}\033[00m".format('---------------- QUERY_COST : $ {cost}---------------- QUERY TOKENS : {tokens}-----------------'.format(cost = round(cb.total_cost, 5) , 
                                                                                                                                             tokens = cb.total_tokens)))
  prediction_df.to_csv(path, index=False)


import pandas as pd

# df1 = pd.read_csv('prediction_data/test_DATA_multi_class_1.csv')
# df2 = pd.read_csv('prediction_data/test_DATA_multi_class_2.csv')
# df3 = pd.read_csv('prediction_data/test_DATA_multi_class_3.csv')
# df4 = pd.read_csv('prediction_data/test_DATA_single.csv')

# df = pd.concat([df1, df2, df3, df4], axis = 0)
# print(df.shape)
# df.to_csv('prediction_data/FINAL_CSV.csv')