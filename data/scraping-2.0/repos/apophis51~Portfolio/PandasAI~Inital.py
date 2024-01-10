import pandas as pd
import pprint
from pandasai import PandasAI

pp = pprint.PrettyPrinter(indent=4)




site_data = pd.read_csv('data.csv')

# print(site_data.head())
# print(site_data)
print(site_data.head())


from pandasai.llm.openai import OpenAI
llm = OpenAI(api_token="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")

pandas_ai = PandasAI(llm)
print(pandas_ai(site_data, prompt='give me a list of all the columns'))
pp.pprint(pandas_ai(site_data, prompt = 'give me a list of the Current Url with the most volume'
))
