import pandas as pd
from pandasai import PandasAI

# Sample DataFrame
df = pd.read_csv("Datasets/the_oscar_award.csv")

# Instantiate a LLM
from pandasai.llm.openai import OpenAI
llm = OpenAI(api_token="YOUR_API_TOKEN")

pandas_ai = PandasAI(llm, conversational=False)
pandas_ai(df, prompt='Which are the 5 happiest countries?')