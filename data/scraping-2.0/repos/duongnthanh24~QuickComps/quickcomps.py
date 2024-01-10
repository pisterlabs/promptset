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
api = st.secrets["api"]
openai.api_key = api
df = pd.read_csv('table_i.csv',encoding = "ISO-8859-1")
e = df["Primary_Industry"].dropna().unique()
industry = '; '.join(e)
initcondition = f"We have a table named df with the following columns: Ticker, Name, Country, Description, Primary_Industry, Detailed_Description, Region, Revenue, COGS, Gross Profit, EBIT, EBITDA, Net_Income, Total_Equity, Market_Capitalization, Revenue_Growth, Revenue_Multiple, EBITDA_Multiple, PE_Ratio, Price_to_book_ratio, ROE. For the Primary_Industry columns, we have the following category: {industry}. For the Country columns, we have the following category: Australia, Austria, Bahamas, Bangladesh, Belgium, Belize, Bermuda, British Virgin Islands, Bulgaria, Canada, Cayman Islands, China, Croatia, Curaçao, Cyprus, Czech Republic, Denmark, Estonia, Falkland Islands, Finland, France, Germany, Gibraltar, Greece, Guernsey, Hong Kong, Hungary, Iceland, India, Indonesia, Iran, Ireland, Isle of Man, Israel, Italy, Ivory Coast, Japan, Jersey, Kazakhstan, Latvia, Liberia, Liechtenstein, Lithuania, Luxembourg, Macedonia, Malaysia, Malta, Marshall Islands, Mauritius, Mexico, Monaco, Mongolia, Netherlands, Netherlands Antilles, New Zealand, Norway, Pakistan, Panama, Papua New Guinea, Philippines, Poland, Portugal, Romania, Russia, Serbia, Singapore, Slovakia, Slovenia, South Africa, South Korea, Spain, Sri Lanka, Sweden, Switzerland, Taiwan, Thailand, Ukraine, United Kingdom, United States, Uruguay, Vietnam. From this point onwards, only reply with an SQL command. You have to take all user content and the latest assistant content into account. First answer should always have 'SELECT Name, Primary_Industry, Country' and you should always check Primary_Industry first. If there is no matching keyword in Primary_Industry, you should check Detailed_Description. "
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system","content": initcondition}]

#####################


########################
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

try:
    #response =  """SELECT Name, Country, Primary_Industry FROM df WHERE Primary_Industry IN ('Insurance','Property and Casualty Insurance') AND (Country = 'China' OR Country = 'Thailand' OR Country = 'South Korea' OR Country = 'Japan')"""
    response = st.session_state.messages[-1]["content"]
    response = "SELECT" + response.split("SELECT")[1]
    #response

    ############# Get a list of columns
    sql_select_list = []
    for i in df.columns:
        if i in response:
            sql_select_list.append(i)
    sql_select = ', '.join(sql_select_list)

    ############
    if "ORDER" in response:
        where = response[response.find("WHERE"):response.find("ORDER")]
        order = response[response.find("ORDER"):]
    else:
        where = response[response.find("WHERE"):]
        order = ""
    
    
    where = where.replace("(","").replace(")","")
    #where

    sql_where_list = []
    for i in where.split():
        if i in df.columns:
            sql_where_list.append(i)

    ind_like = []
    ind_cat = []

    coun_cat = []

    des_like = []

    each_where = []
    for i in range(len(sql_where_list) - 1):
        each_where.append(where[where.find(sql_where_list[i]):where.find(sql_where_list[i + 1])])
    #where
    #where[where.find(sql_where_list[-1]):]
    each_where.append(where[where.find(sql_where_list[-1]):])
    each_where = [x for x in each_where if x != '']
    #each_where
    tail = []

    for x in each_where:
        if x.rfind("OR ") + 3 == len(x):
            x = x[:-3]
        elif x.rfind("AND ") + 4 == len(x):
            x = x[:-4]
        
        if "Primary_Industry" in x:
            s = re.findall(r"'([^']+)'" , x)
            for i in s:
                # if "%" in i:
                #     i=i.replace('%', '')
                #     ind_like.append(i)
                # else:
                #     ind_cat.append(i)
                if "%" in i:
                    i=i.replace('%', '')
                for w in df["Primary_Industry"].unique():
                    if i.lower() in w.lower():
                        ind_cat.append(w)
            
        elif "Country" in x:
            
            s = re.findall(r"'([^']+)'" , x)
            for i in s:
                if "%" in i:
                    i=i.replace('%', '')
                for w in df["Country"].unique():
                    if i.lower() in str(w).lower():
                        coun_cat.append(str(w))

        elif "Detailed_Description" in x:
            s = re.findall(r"'([^']+)'" , x)
            for i in s:
                if "%" in i:
                    i=i.replace('%', '')
                    des_like.append(i)
        else:
            tail.append(x)


    #ind_cat
    coun_cat = list(set(coun_cat))
    ind_cat = list(set(ind_cat))
    #ind_cat
    #ind_cat
    #############Find all quote  items
    # t = re.findall(r"'(.*?)'", response)
    #like = re.findall(r"%(.*?)%", response)

    # industry_matches = []
    # country_matches = []
    # for i in t:
    #     if i in df["Primary_Industry"].unique():
    #         industry_matches.append(i)
    #     elif i in df["Country"].unique():
    #         country_matches.append(i)

     #############Find all like  items
    # industry_pattern2 = r"Primary_Industry\s+LIKE\s+'([^']*)'"  
    # in_inter2 = re.findall(industry_pattern2, response)
    # industry_matches2 = [item.replace('%', '') for item in in_inter2]

    # description_pattern = r"Detailed_Description\s+LIKE\s+'([^']*)'" 
    # de_inter = re.findall(description_pattern, response)
    # description_matches = [item.replace('%', '') for item in de_inter]
    # description_matches

    # des_whole = []
    # des_each = []
    # for i in description_matches:
    #     des_whole.append("Detailed_Description LIKE '%"+i+"%'")
    #     for e in i.split():
    #         des_each.append("Detailed_Description LIKE '%"+e+"%'")
    # #des_each
    # des = " AND ".join(des_each)
    #des

    
    to_filter_columns = st.multiselect("Filter dataframe on", df.columns,sql_select_list)
    column_list = ', '.join(to_filter_columns)
    sql_country = []
    sql_industry_description = []
    des_xx = []
    #sql_description = []

    for column in to_filter_columns:
        left, right = st.columns((1, 20))
        
        if column == "Country":
            if len(coun_cat) > 0:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    coun_cat,
                )
                if len(user_cat_input) == 1:
                    query = "Country IN %s" % repr(tuple(map(str,user_cat_input)))
                    query = query.replace(",","")
                    sql_country.append(query)
                    #query
                    left.write("↳")
                elif len(user_cat_input) > 1:    
                    query = "Country IN %s" % repr(tuple(map(str,user_cat_input)))
                    sql_country.append(query)
                    #query
                    left.write("↳")
                
        if column == "Primary_Industry":
            if len(ind_cat) > 0:
                i_user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    ind_cat,
                )
                #i_user_cat_input
                if len(i_user_cat_input) == 1:
                    query = "Primary_Industry IN %s" % repr(tuple(map(str,i_user_cat_input)))
                    query = query.replace(",","")
                    sql_industry_description.append(query)
                    #query
                    left.write("↳")
                elif len(i_user_cat_input) > 1:    
                    query = "Primary_Industry IN %s" % repr(tuple(map(str,i_user_cat_input)))
                    sql_industry_description.append(query)
                    #query
                    left.write("↳")
            if len(ind_like) > 0:
                i_user_cat_input = right.multiselect(
                    f"Keyword for {column}",
                    ind_like,
                    ind_like,
                )
                query = "Primary_Industry LIKE '%"+i_user_cat_input[0]+"%'"
                sql_industry_description.append(query)
                #query
        if column == "Detailed_Description":
            if len(des_like) > 0:
                i_user_cat_input = right.multiselect(
                    f"Keyword for {column}",
                    des_like,
                    des_like,
                )

                for i in i_user_cat_input:
                    l = i.split()
                    #l
                    if len(l)>1:
                        h = []
                        for r in l:
                            h.append("Detailed_Description LIKE '%"+r+"%'")
                        h = " AND ".join(h)
                        des_xx.append(h)
                    else:
                        des_xx.append("Detailed_Description LIKE '%"+i+"%'")
    des_xx = " AND ".join(des_xx)
    #des_xx
    indus = ' OR '.join(sql_industry_description)
    #des = "("+des+")"
    #indus

    if len(des_xx)>1:
        des_xx = "(" + des_xx + ")"
        if len(indus) > 1:
            e = " OR ".join([des_xx,indus])
        else:
            e = des_xx
    else:
        e = indus

    sql_country = " OR ".join(sql_country)
    #sql_country

    if len(e)>1:
        e = "(" + e + ")"
        f = " AND ".join([e,sql_country])
    else:
        f = sql_country
    #f
    tail = " AND ".join(tail)
    #tail
    if len(tail) > 1:
        g = " AND ".join([f,tail])
    else:
        g = f
    mei = "SELECT "+ column_list +" FROM df WHERE " + g + order
    #response
    #mei
    st.session_state.messages.append({"role": "user", "content": mei})
    st.session_state.messages.append({"role": "assistant", "content": mei})
    
    df2 = ps.sqldf(mei, locals())
    #df2.style.set_sticky(axis="Name")
    df2
    with st.chat_message("user"):
        st.markdown(prompt)
except IndexError:
    pass 
# except:
#     pass