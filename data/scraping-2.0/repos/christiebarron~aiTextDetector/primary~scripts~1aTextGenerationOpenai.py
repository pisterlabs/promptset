# Description #####
#this python script generates text based on a series of prompts using OpenAI's text-davinci-003 model
#It requires an OpenAI API key, which is not included on github.

working_directory = "primary/rawData/aiEssays/"
clean_data_directory = "primary/cleanData/"

#loading depencies #####
#if first time, run pip install openai

from config import OPENAI_API_KEY #loading API key. you will need your own.
import pandas as pd #for data wrangling
import os #for working with paths

import openai #for utilization of large language model
openai.api_key =  OPENAI_API_KEY #save API key

# see   https://platform.openai.com/docs/api-reference/completions/create


# Model Setup ######
model = "text-davinci-003"#the id of the model to include

#create list of prompts
prompt1 = "Write a letter to your local newspaper stating your opinion on the effects computers have on people. Persuade the readers to agree with you." #the prompt to provide 
prompt2 = "Write a persuasive essay to a newspaper about censorship in libraries. Do you believe that certain materials, such as books, music, movies, and magazines should be removed from the shelves if they are found offensive? " #the prompt to provide 
prompt3 = "Write a 200 word response that explains how the features of the setting affect the cyclist in 'Rough Road Ahead' by Joe Kurmaski."
prompt4 = "Why does Winter Hibiscus by Minfong Ho end with 'When they come back in the spring, when the snows melt and the geese return and this hibiscus is budding, then I will take that test again?'" 
prompt5 = "Write a 200 word essay about the mood created in the memoir 'From Home: The Blueprints of Our Lives' by Narciso Rodriguez."
prompt6 = "Write an essay describing the obstacles the builders of the Empire State Building faced in allowing dirigibles to dock. Use information from 'The Mooring Mast' by Marcia Amidon Lüsted."
prompt7 = "Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining. Write a true story about a time someone was patient using first person." #using first person. #the prompt to provide 
prompt8 = "Laughter is the shortest distance between two people. Laughter is important in any relationship. Write a true story involving laughter in first person." #using first person. #the prompt to provide
#prompt9 = ""
#prompt10 = ""
#prompt11 = ""
#prompt12 = ""
#prompt13 = ""
#prompt14 = ""

#prompts = ['prompt{}'.format(i) for i in range(1, 9)] #doesn't work
prompts = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8]

essay_id = 1 #an id of the essay to save to the text file
essay_type = "persuasive/narrative" #type of essay to potentially the save
max_tokens = 300 #maximum number of tokens to include
n = 30 #number of essays to generate in single API call

count = 0 #provide count number as text id to save
count = len(os.listdir(working_directory)) +2 #so don't overwrite previously generated text
# Generating Essays #######
#loop through i API calls


for prompt in prompts:

    #model setup
    model = "text-davinci-003" #the id of the model to include
    #essay_type = "persuasive/narrative" #need to make list-compatible
    #essay_id = i #need to make list-compatible.
    max_tokens = 250 #maximum number of tokens to include
    n = 30 #number of essays to generate in single API call



    for i in range (15): 

        result = openai.Completion.create(model =  model, prompt = prompt, max_tokens = max_tokens, n = n) #API call, keeping result
    
    #loop through x texts generated
        for x in range(len(result["choices"])): 
            count = count + 1
            text = result["choices"][x]["text"] #extract the ai-generated text
            model = result["model"] #extract the model name

        #save the text with a name including the essay id, model, and count
            with open(f'{working_directory}/eid{essay_id}_{model}_{count}.txt', 'w') as f: 
                f.write(text)


#save all the ai essays to a pandas dataframe ######

text_list = [] #create a list to add text files to.
ai_llm = [] #a list of the large language model used
eid = [] #essay prompt id
row_id = [] #final number
files = os.listdir(working_directory) #get a list of all file names within the directory

#extract relevant information from ai-generated text files
for f in files:
    filename = f'{working_directory}{f}' #save the filename

    with open(filename, 'r') as f: #open filename and save the text in it
        text_list.append(f.read())
    
    #append metadata saved in the filename (llm used, essay prompt, and row id)
    ai_llm.append(filename.split("_")[1])
    eid.append(filename.split("_")[0].split("d")[1])
    row_id.append(filename.split("_")[2].split(".")[0])
    
#save all extracted text to a pandas dataframe, then excel file.
df = pd.DataFrame({"row_id" : row_id, "essay_id" : eid, "ai_llm": ai_llm, 'ai_essay': text_list})
df.to_excel("{clean_data_directory}aiGenerated.xlsx")
