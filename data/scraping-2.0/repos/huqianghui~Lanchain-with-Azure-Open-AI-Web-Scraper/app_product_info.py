import pandas as pd
from langchain_stuff_type import get_content_by_index_contents
from json import loads
import os
from resultFinetune import get_finetune_result

# Read the Excel file into a pandas DataFrame
df = pd.read_excel("./documents/GPT需要分析的产品信息.xlsx", sheet_name="产品信息")

# 选择第二列的所有数据
AllIndexData = df.iloc[:, 1].tolist()

jsonResult = loads(get_content_by_index_contents(AllIndexData))

print(jsonResult)

for item in jsonResult:
    index = item["index"]
    value = item["value"]
    df.loc[df['表头信息'] == index, 'gtp值'] = str(value)

# Write the updated DataFrame back to the Excel file
with pd.ExcelWriter("./documents/GPT需要分析的产品信息.xlsx",mode='a',if_sheet_exists='replace') as writer:
    df.to_excel(writer, sheet_name='产品信息', index=False)
