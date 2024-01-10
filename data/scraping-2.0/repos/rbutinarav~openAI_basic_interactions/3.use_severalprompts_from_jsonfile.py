from openai_functions import ai_complete
from general_functions import json_to_df

#EXAMPLE 3: LOAD OF A LIST OF PROMPTS FROM A JSON FILE AND A PREPROMPT FROM A TEXT FILE, USE OF THE ai_complete FUNCTION TO COMPLETE EACH PROMPT AND PRINT OUT A NEW JSON FILE WITH PROMPTS AND COMPLETIONS
'This example shows how to use a list of prompts from a json file and a preprompt from a text file, use of the ai_complete function to complete each prompt and print out a new json file with prompts and completions'
training_reviews = json_to_df('training_reviews.json')
prompt_reviews = training_reviews['prompt']
#print (prompt_reviews)

with open ('prep_reviews.txt', 'r') as f:
    prep = f.read()

for i in prompt_reviews:
    print (i)
    extended_prompt=prep+'\n'+i
    print (ai_complete(extended_prompt, max_tokens=100))
    print ('\n')

