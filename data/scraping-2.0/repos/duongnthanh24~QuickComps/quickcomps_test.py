import openai
import pandas as pd
import pandasql as ps
import streamlit as st
import functools as ft


from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import re

st.set_page_config(
        page_title="Quick Comps Table")
st.header("Quick Comps Table")

api = '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' #st.text_input('Enter API')

openai.api_key = api
df = pd.read_csv('table_i.csv',encoding = "ISO-8859-1")
e = df["Primary_Industry"].dropna().unique()
industry = ', '.join(e)
initcondition = f"We have a table named df with the following columns: Ticker, Name, Country, Description, Primary_Industry, Detailed_Description, Region, Revenue, COGS, Gross Profit, EBIT, EBITDA, Net_Income, Total_Equity, Market_Capitalization, Revenue_Growth, Revenue_Multiple, EBITDA_Multiple, PE_Ratio, Price_to_book_ratio, ROE. For the Primary_Industry columns, we have the following category: {industry}. For the Country columns, we have the following category: Australia, Austria, Bahamas, Bangladesh, Belgium, Belize, Bermuda, British Virgin Islands, Bulgaria, Canada, Cayman Islands, China, Croatia, Curaçao, Cyprus, Czech Republic, Denmark, Estonia, Falkland Islands, Finland, France, Germany, Gibraltar, Greece, Guernsey, Hong Kong, Hungary, Iceland, India, Indonesia, Iran, Ireland, Isle of Man, Israel, Italy, Ivory Coast, Japan, Jersey, Kazakhstan, Latvia, Liberia, Liechtenstein, Lithuania, Luxembourg, Macedonia, Malaysia, Malta, Marshall Islands, Mauritius, Mexico, Monaco, Mongolia, Netherlands, Netherlands Antilles, New Zealand, Norway, Pakistan, Panama, Papua New Guinea, Philippines, Poland, Portugal, Romania, Russia, Serbia, Singapore, Slovakia, Slovenia, South Africa, South Korea, Spain, Sri Lanka, Sweden, Switzerland, Taiwan, Thailand, Ukraine, United Kingdom, United States, Uruguay, Vietnam. From this point onwards, only reply with an SQL command. You have to take all user content and the latest assistant content into account. First answer should always have 'SELECT Name, Primary_Industry, Country' and you should always check Primary_Industry first. If there is no matching keyword in Primary_Industry, you should check Detailed_Description. "

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system","content": initcondition}]


a = ["a"]
if prompt := st.chat_input("Top 10 banks in Vietnam by Revenue"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    messages = st.session_state.messages
    #if len(st.session_state.messages) > 1:
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = st.session_state.messages
        )
    response = response.choices[0].message.content.strip()

    a.append(response)
    #response

    st.session_state.messages.append({"role":"assistant","content": response})
    # df2 = ps.sqldf(response, locals())
    # if len(df2) == 0:
    #     with st.chat_message("assistant"):
    #         st.markdown("recalibrating...")
    #     st.session_state.messages.append({"role": "user", "content": "You also need to check Primary_Industry OR Detailed_Description. Make sure to include 'SELECT Name, Primary_Industry, Country, Description' in your answer."})
    #     response = openai.ChatCompletion.create(
    #         model = "gpt-3.5-turbo",
    #         messages = st.session_state.messages
    #         )
    #     response = response.choices[0].message.content.strip()
    #     #response
    #     st.session_state.messages.append({"role":"assistant","content": response})

try:
    response = st.session_state.messages[-1]["content"]
    response = "SELECT" + response.split("SELECT")[1]
    response
    

    sql_select_list = []
    for i in df.columns:
        if i in response:
            sql_select_list.append(i)
    sql_select = ', '.join(sql_select_list)
    response = response.replace(response[response.find("SELECT")+7:response.find("FROM")-1], sql_select)

    country_pattern = r"Country\s*=\s*'([^']+)'" 
    country_matches = re.findall(country_pattern, response)
    industry_pattern = r"Primary_Industry\s*=\s*'([^']+)'" 
    in_inter = re.findall(industry_pattern, response)

    industry_matches = []
    for i in in_inter:
        if i in df["Primary_Industry"].unique():
            industry_matches.append(i)

    industry_pattern2 = r"Primary_Industry\s+LIKE\s+'([^']*)'"  
    in_inter2 = re.findall(industry_pattern2, response)
    industry_matches2 = [item.replace('%', '') for item in in_inter2]

    description_pattern = r"Detailed_Description\s+LIKE\s+'([^']*)'" 
    de_inter = re.findall(description_pattern, response)
    description_matches = [item.replace('%', '') for item in de_inter]

    df = df.copy()
    modification_container = st.container()
    with modification_container:

        to_filter_columns = st.multiselect("Filter dataframe on", df.columns,sql_select_list)
        lvl1 = df
        for column in to_filter_columns:

            left, right = st.columns((1, 20))
            
            if column == "Country":
                if len(country_matches) > 0:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        country_matches,
                    )
                    lvl1 = lvl1[lvl1[column].isin(user_cat_input)]
                    left.write("↳")
                    
            if column == "Primary_Industry":
                if len(industry_matches) > 0:
                    i_user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        industry_matches,
                    )
                    lvl1 = lvl1[lvl1[column].isin(i_user_cat_input)]
                    left.write("↳")
                else:
                    i_user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                    )
                    #lvl1 = lvl1[lvl1[column].isin(i_user_cat_input)]
                    left.write("↳")
                if len(industry_matches2) > 0:
                    user_cat_input = right.multiselect(
                        f"Keyword for {column}",
                        industry_matches2,
                        industry_matches2,
                    ) 
                    #lvl1 = lvl1[lvl1[column].str.contains(industry_matches2[0],na=False)]
                    left.write("↳")

            if column == "Detailed_Description":
                if len(description_matches) > 0:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        description_matches,
                        description_matches,
                    ) 
                    lvl1 = lvl1[lvl1[column].str.contains(description_matches[0],na=False)]
                    left.write("↳")

            # elif is_numeric_dtype(df[column]):
            #     _min = float(df[column].min())
            #     _max = float(df[column].max())
            #     step = (_max - _min) / 100
            #     user_num_input = right.slider(
            #         f"Values for {column}",
            #         min_value=_min,
            #         max_value=_max,
            #         value=(_min, _max),
            #         step=step,
            #     )
            #     num = df[df[column].between(*user_num_input)]
        #lvl1 = pd.merge(coun, ind, how="inner")

        
        #lvl1 = ft.reduce(lambda left, right: pd.merge(left, right, on='Name'), dfs)

        #if des:
        #    lvl1 = pd.merge(pd.merge(lvl1,des,on='Name'))
                  
        lvl1 = lvl1[to_filter_columns]
        #df.sort_values(by=['col1']).head(10)
        #limit 
        a = response[response.find("df"):-1] #+ response[-1]
        a = a.split()
        if "DESC" in a:
            b = a[a.index("DESC")-1]
            lvl1 = lvl1.sort_values(by=[b],ascending=False)
        if "LIMIT" in a:
            b = a[a.index("LIMIT")+1]
            lvl1 = lvl1.head(int(b))

        t = ', '.join(to_filter_columns)
        response = response.replace(response[response.find("SELECT")+7:response.find("FROM")-1], t)
        st.session_state.messages.append({"role":"assistant","content": response})
    lvl1
    with st.chat_message("user"):
        st.markdown(prompt)
except IndexError:
    pass





