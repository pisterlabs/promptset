import openai
import pandas as pd
import pandasql as ps
import streamlit as st
import functools as ft
import numpy as np

from streamlit_tags import st_tags


from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import re
api = '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' #st.text_input('Enter API')
openai.api_key = api
df = pd.read_csv('table_i.csv',encoding = "ISO-8859-1")

response = """SELECT Name, Country, Description, Primary_Industry, Detailed_Description FROM df WHERE ((Detailed_Description LIKE '%coffee%' AND Detailed_Description LIKE '%chain%') OR Primary_Industry LIKE '%coffee chain%') AND Country IN ('China', 'South Korea')"""
df2 = ps.sqldf(response, locals())
df2 


"## Bold words in a table"
needle = "coffee"
stack = df2["Detailed_Description"]
#pretty_stack = re.sub(f"({needle})",r"**\1**",stack,flags=re.IGNORECASE)
e = []
for i in df2["Detailed_Description"]:
    e.append(re.sub(f"({needle})",r"**\1**",i,flags=re.IGNORECASE))
e
df2["Detailed_Description"] = e

st.markdown(df2.to_markdown(None))


where = response[response.find("WHERE"):response.find("ORDER")]
order = response[response.find("ORDER"):]
where = where.replace("(","").replace(")","")

sql_where_list = []
for i in where.split():
    if i in df.columns:
        sql_where_list.append(i)

#sql_where_list.append('ORDER')

ind_like = []
ind_cat = []

coun_cat = []

des_like = []

each_where = []
for i in range(len(sql_where_list) - 1):
    each_where.append(where[where.find(sql_where_list[i]):where.find(sql_where_list[i + 1])])
each_where.append(where[where.find(sql_where_list[-1]):])
each_where
each_where[2]
w = each_where[2].rfind("OR ") +3
w
w= len(each_where[2])
w
each_where[2][:-3]

for x in each_where:
    if x.rfind("OR ") + 3 == len(x) or x.rfind("AND ") + 4 == len(x):
        x = x[:-3]
    if "Primary_Industry" in x:
        s = re.findall(r"'([^']+)'" , x)
        for i in s:
            if "%" in i:
                ind_like.append(i)
            else:
                ind_cat.append(i)
    elif "Country" in x:
        s = re.findall(r"'([^']+)'" , x)
        for i in s:
            if "%" not in i:
                coun_cat.append(i)
    elif "Detailed_Description" in x:
        s = re.findall(r"'([^']+)'" , x)
        for i in s:
            if "%" in i:
                des_like.append(i)

ind_like
ind_cat
coun_cat
des_like