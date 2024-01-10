import pandas as pd
import openai
from dotenv import load_dotenv
import tiktoken
import os

# I have scraped the following University Data from online sources that I want you to rephrase and remove entire plagiarism with very specific instructions.

# Firstly, here are the parameters that I want you to rephrase: 
# <Name the paramters>

# Instructions: 

# 1. Your rephrased output should be of the exact same length as the input data. 
# 2. Humanize the data and remove any plagiarism since we are using this data on our website. 
# 3. We are using this for strong SEO purposes, hence make use of SEO oriented rephrasing language with maximum keywords.
i = 1


load_dotenv()

openai.api_key = os.getenv("API_KEY")

column_to_enhance = ['Overview','Eligibility Overview','Course Ranking','Placement','University Ranking','Accommodation','Course Description']
csv_file_path = 'University4.csv'
df = pd.read_csv(csv_file_path)


def num_tokens_from_string(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



def enhance_text(text):
    global i
    if pd.notna(text):
        try:
            tokens = num_tokens_from_string(text, "gpt-3.5-turbo-instruct")
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=f"""{text}\n\n This is some data for a university. I want you to rephrase and remove entire plagiarism with very specific instructions.
                                    1. Your rephrased output should be of the exact same length as the input data. 
                                    2. We are using this for strong SEO purposes, hence make use of SEO oriented rephrasing language with maximum keywords.""",
                max_tokens=tokens + 100,
            )
            print(f"Response Generated Successfully {i}")
            i += 1
            return response.choices[0].text.strip()
        except:
            print("Error")
            i += 1
            return "Error"
    else:
        print("No Text was provided")
        i += 1
        return "No text provided"



def enhance_and_track_changes(dataframe, column_names):
    for column in column_names:
        dataframe[column] = dataframe[column].apply(enhance_text)
        print(f"{column} enhanced")
    return dataframe



df = enhance_and_track_changes(df, column_to_enhance)
enhanced_csv_path = 'Enhanced_University4.csv'
df.to_csv(enhanced_csv_path, index=False)

print(f"Enhanced data with original values saved to {enhanced_csv_path}")


