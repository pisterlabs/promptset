#%% Import
import pandas as pd
import re
import openai
import os
from openai.embeddings_utils import get_embedding
import unicodedata
from unidecode import unidecode

# Load your API key from an environment variable or secret management service
openai.api_key = "YOUR_OPENAI_API_KEY"
#openai.api_key = ""

### ---------------------------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------------------------
### ---------------------------------------------------------------------------------------------------------

######################### CLEANING MANUAl AND GLOSSARY DATAFRAMES #########################
#%% Read the CSVs into the code
manual = pd.read_csv("M28C_Scrap.csv")
glossary = pd.read_csv("GlossaryTerms.csv")
extra = pd.read_csv("ExtraData.csv")

#%% Functions to clean the CSVs
# Function that removes leading whitespace
def remove_extra_spaces(text):
    fixed_text = text.lstrip()
    return fixed_text

# Function for apostrophes and double quotes (encoding issue)
def standardize_chars(text):
    fixed_text = unidecode(text).replace("\\", "").replace("’", "'")
    return fixed_text

# Function to standardize heading numbers (removes extra period)
def standardize_heading(heading):
    # Match a double number followed by a period and an extra period
    pattern = r"(\d+\.\d+)\."
    # Replace the second period with an empty string
    cleaned_heading = re.sub(pattern, r"\1", heading)
    return cleaned_heading

# Function to format the headings with the same spacing (all headings should be X.XX  Chapter Name)
def format_heading(heading):
    # Match a double number followed by one or more spaces and a word
    pattern = r"(\d+\.\d+)\s{2}(\w+|\d+)"
    # Check if the heading matches the expected format
    match = re.match(pattern, heading)
    if match:
        # Heading is already in the correct format
        return heading
    else:
        # Extract the first two numbers and the remaining text
        match = re.match(r"(\d+\.\d+)\s*(.*)", heading)
        if match:
            # Check if there's only one space after the number
            if len(match.group(2)) > 0 and match.group(2)[0] != " ":
                # Add an additional space after the number
                formatted_heading = f"{match.group(1)}  {match.group(2)}"
            else:
                # Heading is already in the correct format
                formatted_heading = heading
            return formatted_heading
        else:
            # Heading doesn't match any expected format, return it as is
            return heading

#%% Prepare the GlossaryTerms CSV for embedding
# Step 1: Fix unicode
glossary['Content'] = glossary['Content'].apply(standardize_chars)
glossary['Heading'] = glossary['Heading'].apply(standardize_chars)
glossary['Chapter Title'] = glossary['Chapter Title'].apply(standardize_chars)
# Step 2. Remove leading whitespace
glossary['Content'] = glossary['Content'].apply(remove_extra_spaces)
glossary['Heading'] = glossary['Heading'].apply(remove_extra_spaces)
glossary['Chapter Title'] = glossary['Chapter Title'].apply(remove_extra_spaces)
#glossary.to_csv('Example.csv', index=False)

#%% Prepare the Manual CSV for embedding
# Step 1: Fix unicode
manual['Chapter Title'] = manual['Chapter Title'].apply(standardize_chars)
manual['Heading'] = manual['Heading'].apply(standardize_chars)
manual['Content'] = manual['Content'].apply(standardize_chars)
# Step 2: Standardize Headings (Extra period(s) removed)
manual['Heading'] = manual['Heading'].apply(standardize_heading)
# Step 3: Format Headings (spacing fixed)
manual['Heading'] = manual['Heading'].apply(format_heading)
#manual.to_csv('Example.csv', index=False)

#%% Prepare the ExtraData CSV for embedding
extra['Chapter Title'] = extra['Chapter Title'].apply(standardize_chars)
extra['Heading'] = extra['Heading'].apply(standardize_chars)
extra['Content'] = extra['Content'].astype(str)
extra['Content'] = extra['Content'].apply(standardize_chars)
extra = extra.drop('Question to Answer', axis=1)
#extra.to_csv('Example.csv', index=False)

#%% Combine Manual dataframe and Glossary dataframe into one dataframe
frames = [manual, glossary, extra]
df = pd.concat(frames)
# df.to_csv('Example.csv', index=False)

#%% Create a context column by concatenating the title, heading, and the content of each section
# To be used for creating questions and answers below
df = df.rename(columns={'Chapter Title': 'Title'})
df['Context'] = df.Title + "\n" + df.Heading + "\n\n" + df.Content

#%% Save CSV to be used in Embedding.py
df.to_csv("ForEmbedding.csv", index=False)

### -----------------------------------------------------------------------------------
### -----------------------------------------------------------------------------------
### -----------------------------------------------------------------------------------

############### CREATING QUESTIONS AND ANSWERS FROM CONTEXT USING OPENAI ###############
#%% Test the API connection
#response = openai.Completion.create(
 #   engine="text-davinci-001",
  #  prompt="Test prompt",
   # max_tokens=5
#)
# Check the response
#print(response)

#%% Create questions based on the context
# Can experiment more with temperature, max_tokens, top_p, frequency_penalty, and presence_penalty
#def get_questions(Context):
 #   try:
  #      response = openai.Completion.create(
   #         engine="text-davinci-001",
    #        prompt=f"Write questions based on the text below\n\nText: {Context}\n\nQuestions:\n1.",
     #       temperature=0,
      #      max_tokens= 300,
       #     top_p=1,
        #    frequency_penalty=0,
         #   presence_penalty=0,
          #  stop=["\n\n"]
        #)
        #return response['choices'][0]['text']
    #except:
     #   return ""

#df['Questions']= df.Context.apply(get_questions)
#df['Questions'] = "1." + df.Questions
#print(df[['Questions']].values[0][0])

# %% Check if questions populated correctly
#df.to_csv('M28C_Q.csv', index=False)

#%% Create answers based on the context and question
#  Nearly 2 hour runtime (Increase tokens to 300 when have time)
#def get_answers(row):
 #   try:
  #      response = openai.Completion.create(
   #         engine="text-davinci-001",
    #        prompt=f"Write answer based on the text below\n\nText: {row.Context}\n\nQuestions:\n{row.Questions}\n\nAnswers:\n1.",
     #       temperature=0,
      ##     top_p=1,
        #    frequency_penalty=0,
         #   presence_penalty=0
        #)
        #return response['choices'][0]['text']
    #except Exception as e:
     #   print (e)
      #  return ""

#df['Answers']= df.apply(get_answers, axis=1)
#df['Answers'] = "1." + df.Answers
#df = df.dropna().reset_index().drop('index',axis=1)
#print(df[['Answers']].values[0][0])

#%% Check if questions and answers populated correctly
#df.to_csv('M28C_QA.csv', index=False)
