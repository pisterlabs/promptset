import openai
import pandas as pd
import pandasql as ps
import streamlit as st
import functools as ft
import plotly.express as px


df = pd.read_csv('table_g.csv',encoding = "ISO-8859-1")
e = df["Primary_Industry"].dropna().unique()

industry = ', '.join(e)
#df = a.head(20)
#df

exec("df_sorted = df.sort_values('Market_Capitalization', ascending=False)")
exec("fig = px.bar(df_sorted[:10], x='Name', y='Market_Capitalization', title='Top 10 Market Cap')")
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

exec("df_vietnam_banks = df[df['Primary_Industry'] == 'Banks']")
exec("fig = px.bar(df_vietnam_banks, x='Name', y='Market_Capitalization', title='Top Banks in Vietnam by Market Capitalization')")
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

api = st.text_input('Enter API')
openai.api_key = api


initcondition = f"""
We have a table named df with the following columns: Name, Country, Description, Primary_Industry, Detailed_Description, Region, Revenue, COGS, Gross Profit, EBIT, EBITDA, Net_Income, Total_Equity, Market_Capitalization, Revenue_Growth, Revenue_Multiple, EBITDA_Multiple, PE_Ratio, Price_to_book_ratio, ROE
For the Primary_Industry columns, we have the following category: {industry}.
For the Country columns, we have the following category: Australia, Austria, Bahamas, Bangladesh, Belgium, Belize, Bermuda, British Virgin Islands, Bulgaria, Canada, Cayman Islands, China, Croatia, Curaçao, Cyprus, Czech Republic, Denmark, Estonia, Falkland Islands, Finland, France, Germany, Gibraltar, Greece, Guernsey, Hong Kong, Hungary, Iceland, India, Indonesia, Iran, Ireland, Isle of Man, Israel, Italy, Ivory Coast, Japan, Jersey, Kazakhstan, Latvia, Liberia, Liechtenstein, Lithuania, Luxembourg, Macedonia, Malaysia, Malta, Marshall Islands, Mauritius, Mexico, Monaco, Mongolia, Netherlands, Netherlands Antilles, New Zealand, Norway, Pakistan, Panama, Papua New Guinea, Philippines, Poland, Portugal, Romania, Russia, Serbia, Singapore, Slovakia, Slovenia, South Africa, South Korea, Spain, Sri Lanka, Sweden, Switzerland, Taiwan, Thailand, Ukraine, United Kingdom, United States, Uruguay, Vietnam.
I will ask you to create a chart named fig.
From this point onwards, ONLY reply with plotly code. Only the code, no explanation. You do not need to import pandas or plotly because we have already imported them.
Do not include 'fig.show()' in your answer!
You have to take all user content and the latest assistant content into account.
"""
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system","content": initcondition}]

if prompt := st.chat_input("Top 10 banks in Vietnam by Revenue"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    messages = st.session_state.messages
    #if len(st.session_state.messages) > 1:
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = st.session_state.messages
        )
    response = response.choices[0].message.content.strip()
    #response
    st.session_state.messages.append({"role":"assistant","content": response})

    if 'import' in response:
        response = "import" + response.split("import")[1]

    if 'fig' in response:
        response = response.splitlines()
        for i in response:
            exec(i)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    with st.chat_message("user"):
        st.markdown(prompt)

    # if 'df' in response:
    #     response = "df =" + response
    #     # response = response.splitlines()
    #     exec(response)
    #     st.dataframe(df)

    #response
    #response = response.splitlines()
    #response = response[0:-1]
    # response
    # for i in response:
    #     exec(i)

    
    #    st.dataframe()
    #a = df[df['Primary_Industry'] == 'Banks'][df['Country'] == 'Vietnam'].nlargest(10, 'Market_Capitalization')[['Name', 'Market_Capitalization']]
    #a
    #st.dataframe(a)
    
    
    #st.session_state.messages