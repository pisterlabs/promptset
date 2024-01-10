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
        page_title="M&A Comps Table")
st.header("M&A Comps Table")

api = st.text_input('Enter API')

openai.api_key = api
df = pd.read_csv('precedent3.csv',encoding = "ISO-8859-1")
sector = df["Target_Sector"].unique()
set(sector)

# unique_col = df.columns
# unique_col = ', '.join(unique_col)
# unique_col
sector = ', '.join(sector)

tc = df["Target_Country"].unique()
bc = df["Bidder_Country"].unique()
resultList= list(set(tc) | set(bc))


#country = df["Target_Country"].unique()
country = ', '.join(resultList)

initcondition = f"""
We have a table named df with the following columns: Announced_Date, Target_Company, Bidder_Company, Target_Description, Bidder_Description, Target_Country, Bidder_Country, Target_Sector, Bidder_Sector, Target_Website, Bidder_Website, Target_City, Seller_Company, Seller_Description, Seller_Website, Seller_Sector, Seller_Country, Implied_Equity_Value_m, Currency, Net_Debt_m, Enterprise_Value_m, Reported_Y1_Date, Reported_Revenue_m_Y1, Reported_EBITDA_m_Y1, Reported_EBIT_m_Y1, Reported_Earnings_m_Y1, Reported_Earnings_Per_Share_Y1, Reported_Book_Value_m_Y1, Reported_Revenue_Multiple_Y1, Reported_EBIT_Multiple_Y1, Reported_EBITDA_Multiple_Y1, Reported_PE_Multiple_Y1, Reported_Book_Value_Multiple_Y1, Total_Equity_Funding, Deal_Description, Deal_Value_USDm.

List of sector: {sector} .
List of country: {country} .

From this point onwards, only reply with an SQL command. You have to take all user content and the latest assistant content into account.

Only 'SELECT' the column that you use. Never SELECT all column.
Consider searching in Bidder_Description or Target_Description description if you cannot find the appropriate Sector'
"""


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system","content": initcondition}]


a = ["a"]
if prompt := st.chat_input("Top 10 banks in Vietnam by Revenue"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    messages = st.session_state.messages
    #if len(st.session_state.messages) > 1:
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = st.session_state.messages
        )
    response = response.choices[0].message.content.strip()

    a.append(response)
    response

    st.session_state.messages.append({"role":"assistant","content": response})
    df2 = ps.sqldf(response, locals())
    df2
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

# try:
#     response = st.session_state.messages[-1]["content"]
#     response = "SELECT" + response.split("SELECT")[1]
    

#     sql_select_list = []
#     for i in df.columns:
#         if i in response:
#             sql_select_list.append(i)
#     sql_select = ', '.join(sql_select_list)
#     response = response.replace(response[response.find("SELECT")+7:response.find("FROM")-1], sql_select)

#     country_pattern = r"Country\s*=\s*'([^']+)'" 
#     country_matches = re.findall(country_pattern, response)
#     industry_pattern = r"Primary_Industry\s*=\s*'([^']+)'" 
#     in_inter = re.findall(industry_pattern, response)
#     industry_matches = []
#     for i in in_inter:
#         if i in df["Primary_Industry"].unique():
#             industry_matches.append(i)

#     description_pattern = r"Detailed_Description\s+LIKE\s+'([^']*)'" 
#     de_inter = re.findall(description_pattern, response)
#     description_matches = [item.replace('%', '') for item in de_inter]

#     df = df.copy()
#     modification_container = st.container()
#     with modification_container:

#         to_filter_columns = st.multiselect("Filter dataframe on", df.columns,sql_select_list)
#         lvl1 = df
#         for column in to_filter_columns:

#             left, right = st.columns((1, 20))
            
#             if column == "Country":
#                 if len(country_matches) > 0:
#                     user_cat_input = right.multiselect(
#                         f"Values for {column}",
#                         df[column].unique(),
#                         country_matches,
#                     )
#                     lvl1 = lvl1[lvl1[column].isin(user_cat_input)]
#                     left.write("↳")
                    
#             if column == "Primary_Industry":
#                 if len(industry_matches) > 0:
#                     i_user_cat_input = right.multiselect(
#                         f"Values for {column}",
#                         df[column].unique(),
#                         industry_matches,
#                     )
#                     lvl1 = lvl1[lvl1[column].isin(i_user_cat_input)]
#                     left.write("↳")

#             if column == "Detailed_Description":
#                 if len(description_matches) > 0:
#                     user_cat_input = right.multiselect(
#                         f"Values for {column}",
#                         description_matches,
#                         description_matches,
#                     ) 
#                     lvl1 = lvl1[lvl1[column].str.contains(description_matches[0],na=False)]
#                     left.write("↳")

                  
#         lvl1 = lvl1[to_filter_columns]

#         a = response[response.find("df"):-1] 
#         a = a.split()
#         if "DESC" in a:
#             b = a[a.index("DESC")-1]
#             lvl1 = lvl1.sort_values(by=[b],ascending=False)
#         if "LIMIT" in a:
#             b = a[a.index("LIMIT")+1]
#             lvl1 = lvl1.head(int(b))

#         t = ', '.join(to_filter_columns)
#         response = response.replace(response[response.find("SELECT")+7:response.find("FROM")-1], t)
#         st.session_state.messages.append({"role":"assistant","content": response})
#     lvl1
# except IndexError:
#     pass





