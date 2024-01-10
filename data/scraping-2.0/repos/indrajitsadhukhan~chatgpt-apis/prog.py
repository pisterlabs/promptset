import openai
import pandas as pd
input_file="SampleData.csv"
data=pd.read_csv(input_file)

openai.api_key=""

#convert the dataframe to a string
data_str = "The dataframe is :\n"+data.to_string()

question = "\nPrint the sales representation names in decreasing order of total sales"
prompt = data_str + question



# send the dataframe to the ChatGPT model
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=(prompt),
    temperature=0,
    max_tokens=2000,
    top_p=1,
    frequency_penalty=1,
    presence_penalty=1
)
print(response["choices"][0]["text"])
