'''
step 1
read test.txt file
create the dataframe for the llm output
model_names = ["llama2_7b", "mistral_7b"]
'''

import numpy as np
import pandas as pd

model_names = ["llama2_7b", "mistral_7b"] #dont change this!

def generate_csv(model_name):
    # Path to the test.txt file
    test_file_path = 'data/test.txt'
    data = pd.read_table(test_file_path)
    data['llm_labels'] = np.nan
    csv_file_path=f"output/llm_output_{model_name}.csv"
    data.to_csv(csv_file_path, sep='\t', index=False)

############################################################################################################################## 

'''
step 2
read llm_output.csv file (to ensure scalability and run test.txt in batches)
input the sentences into the function `get_output_label`
output the labels
update the llm_output.csv file
'''
from open_source_llm import get_output_dict_llama, get_output_dict_mistral
import langchain 

#check data index start and end
def get_rows(model_name):
    csv_file_path=f"output/llm_output_{model_name}.csv"
    data = pd.read_csv(csv_file_path, sep='\t')
    num_rows = len(data)
    return num_rows


# scalable function to run it in batches
def llm_outputs(start_index,end_index,model_fn,model_name):
    # read csv file as dataframe
    csv_file_path=f"output/llm_output_{model_name}.csv"
    data = pd.read_csv(csv_file_path, sep='\t')
    llm_labels = []
    for sentence in data['Text'].values[start_index:end_index]: #can edit which rows i wanna run
        x = model_fn(sentence=sentence)
        llm_labels.append(int(x['label']))
    data.loc[start_index:end_index-1, 'llm_labels'] = llm_labels
    data.to_csv(csv_file_path, sep='\t', index=False)
    start, end = start_index+2, end_index+1
    return f"saved llm_outputs for row {start} to {end} into {csv_file_path}"

############################################################################################################################## 

if __name__ == '__main__':
    generate_csv(model_names[0])
    generate_csv(model_names[1]) # completed full run already