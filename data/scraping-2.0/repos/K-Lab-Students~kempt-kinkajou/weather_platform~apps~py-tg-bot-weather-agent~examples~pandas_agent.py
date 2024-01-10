import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent



OPENAI_API = "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"

df = pd.read_csv('./files/extracted_data.csv')
print(df)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, openai_api_key=OPENAI_API)

agent = create_pandas_dataframe_agent(chat, df, verbose=True)

msg = agent.run("What anomalies exist in the data?")
print(msg)
