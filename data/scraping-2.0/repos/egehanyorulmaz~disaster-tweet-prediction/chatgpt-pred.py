import pandas as pd
import os
import time

from langchain.llms import OpenAI
from langchain import PromptTemplate

openai_api_key=''

cwd = os.getcwd()
print("Current working directory:", cwd)

df = pd.read_csv(cwd + '/data/test.csv')

llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)


# Notice "location" below, that is a placeholder for another value later
template = """
Classify the following tweet whether it is about disaster or not. 
Your answer must be only 0 for non-disaster 1 for disaster. 
Just say 0 or 1. Tweet: "{tweet}"
"""

tweet_class_cache = []
iteration = df.shape[0]

# create a txt file to store the answers, so that we can use it later
f = open("llm_answers.txt", "w")

for index in range(iteration):
    prompt = PromptTemplate(
        input_variables=["tweet"],
        template=template
    )

    final_prompt = prompt.format(tweet=df['text'][index])
    try:
        tweet_class = int(llm(final_prompt))
        tweet_class_cache.append(tweet_class)
        # write the answer to the txt file with the dataframe index
        f.write(str(index) + ":" + str(tweet_class) + "\n")
        time.sleep(.02)
        print(str(index) + ":" + str(tweet_class))

    except:
        print("Error in index: ", index)
f.close()


# read the txt file with the answers
f = open("llm_answers.txt", "r")
lines = f.readlines()
f.close()

# remove \n from the elements of lines
lines = [line.strip().split(":") for line in lines]

# create a dataframe with two columns, one for the index and one for the answer
df = pd.DataFrame(lines, columns=["id","target"])
df.to_csv("data/submission_gpt.csv", index=False)