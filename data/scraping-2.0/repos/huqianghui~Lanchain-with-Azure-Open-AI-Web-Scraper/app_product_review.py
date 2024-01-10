import pandas as pd
import os
from langchain_refine_type import get_keywords,get_pros_cons_disadvantages
from json import loads

# clear file path,use reivew folder
os.environ['DOC_FILE_PATH'] = ''

# Read the Excel file into a pandas DataFrame
df = pd.read_excel("./documents/GPT需要分析的产品信息.xlsx", sheet_name="产品风评")

# 选择第二列的keywords行数据
AllKeywordsIndexData = df.iloc[0:10, 1].tolist()

jsonKeywordsResult = loads(get_keywords(AllKeywordsIndexData))

# 选择第二列的产品优劣势行数据
proConsIndexData = df.iloc[10:, 1].tolist()

jsonProConsResult = loads(get_pros_cons_disadvantages(proConsIndexData))

jsonResult = jsonKeywordsResult + jsonProConsResult

for item in jsonResult:
    index = item["index"]
    value = item["value"]
    df.loc[df['表头信息'] == index, 'gtp值'] = str(value)

# Write the updated DataFrame back to the Excel file
with pd.ExcelWriter("./documents/GPT需要分析的产品信息.xlsx",mode='a',if_sheet_exists='replace') as writer:
    df.to_excel(writer, sheet_name='产品风评', index=False)