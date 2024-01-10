from open_source_llm import get_output_dict
import pandas as pd
import langchain 
import numpy as np

'''
ollama pull llama2
read llm_output.csv file (to ensure scalability and run test.txt in batches)
input the sentences into the function `get_output_label`
output the labels
update the llm_output.csv file
'''

# read csv file as dataframe
csv_file_path = 'data/llm_output.csv'
data = pd.read_csv(csv_file_path, sep='\t')

llm_labels = []
start_index = 242
end_index = 245

for sentence in data['Text'].values[start_index:end_index]: #can edit which rows i wanna run
    x = get_output_dict(sentence=sentence)
    llm_labels.append(int(x['label']))
        # explanation.append(x['explanation'])
    # except langchain.OutputParserException: 
    #     print("OutputParserException from langchain")
    # except Exception as e:
    #     print(f'An unexpected error occurred: {e}')
    #     x = 2
    #     llm_labels.append(int(x['label']))


# print(llm_labels)
data.loc[start_index:end_index-1, 'llm_labels'] = llm_labels
# data['llm_labels'] = data['llm_labels'].astype('int') # do this when everything done
# data['reasoning'] = explanation
data.to_csv('data/llm_output.csv', sep='\t', index=False)
# print(data.head())


