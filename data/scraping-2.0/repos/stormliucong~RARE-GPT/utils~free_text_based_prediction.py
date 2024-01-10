# openai chat complete example
import os
import openai
import json
import requests
import pandas as pd

# read json file
with open('./api_key.json') as f:
  data = json.load(f)
  openai.api_key = data['openai_token']

def generate_gpt4_response(content, print_output=False):
  '''
  Generates a response from the GPT-4 chatbot given a prompt.
  '''

  completions = openai.ChatCompletion.create( #a method that allows you to generate text-based chatbot responses using a pre-trained GPT language model.
      model="gpt-4", 
      temperature = 0, #controls the level of randomness or creativity in the generated text; . A higher temperature value will result in a more diverse and creative output, as it increases the probability of sampling lower probability tokens. 
#         max_tokens = 2000, #controls the maximum number of tokens (words or subwords) in the generated text.
#         stop = ['###'], #specifies a sequence of tokens that the GPT model should stop generating text when it encounters
      n = 1, #the number of possible chat completions or responses that the GPT model should generate in response to a given prompt
      messages=[
        {'role':'user', 'content': content},
        ])
  
  # return status code

  # Displaying the output can be helpful if things go wrong
  if print_output:
      print(completions)

  # Return the first choice's text
  return completions.choices[0]['message']['content'] #I only want the first repsonses

dx_dict = {}
data_folder = './Data/free_text_input'

free_text_df = pd.read_csv(os.path.join(data_folder, 'free_text_pmid_input.csv'))

free_text_df_subset = free_text_df.iloc[0:2] # for test purpose.
free_text_df_subset = free_text_df # for test purpose.

for index, row in free_text_df_subset.iterrows():
    free_text = row['Free-text']
    id = row['ID']
    dx_gene = row['Gene']
    seq = str(row['Sequence'])
    for top_n in ['5', '10', '50']:
        
        output_path = os.path.join('.', 'Data', 'free_text_input','GPT_response', 'top_' + top_n, id + '_' + seq)
        output_error_path = os.path.join('.', 'Data', 'free_text_input','GPT_response', 'top_' + top_n, id + '_' + seq + '_error')

        if not os.path.exists(os.path.join('.', 'Data', 'free_text_input', 'GPT_response', 'top_' + top_n)):
          os.makedirs(os.path.join('.', 'Data', 'free_text_input','GPT_response', 'top_' + top_n))
        # file exists
        if not os.path.exists(output_path):  
            # call api to get gene prioritization
            content = 'The phenotype desciprtion of the patient is {content}. Can you suggest a list of top {x} possible genes to test? Please return gene symbols as a comma separated list. Example: "ABC1,BRAC2,BRAC1"'.format(x = top_n, content = free_text)
            print(content)
            try:
                gene_prioritization = generate_gpt4_response(content,print_output=True)
                with open(output_path, 'w') as f:
                    f.write(gene_prioritization)
            except Exception as e:
                gene_prioritization = str(e)
                with open(output_error_path, 'w') as f:
                    f.write(gene_prioritization)
          
    

