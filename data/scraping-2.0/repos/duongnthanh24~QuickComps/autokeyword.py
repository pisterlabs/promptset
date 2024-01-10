import openai
import pandas as pd
import pandasql as ps
import streamlit as st


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

api = st.text_input('Enter API')

openai.api_key = api

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system","content": "We have a table named df with the following columns: Ticker, Name, Country, Description, Primary_Industry, Detailed_Description, Region, Revenue, COGS, Gross Profit, EBIT, EBITDA, Net_Income, Total_Equity, Market_Capitalization, Revenue_Growth, Revenue_Multiple, EBITDA_Multiple, PE_Ratio, Price_to_book_ratio, ROE. For the Primary_Industry columns, we have the following category: Advertising, Aerospace and Defense, Agricultural and Farm Machinery, Agricultural Products and Services, Air Freight and Logistics, Airport Services, Alternative Carriers, Aluminum, Apparel Retail, Apparel, Accessories and Luxury Goods, Application Software, Asset Management, Automobile Manufacturers, Automotive Parts and Equipment, Automotive Retail, Biotechnology, Brewers, Broadcasting, Broadline Retail, Building Products, Cable and Satellite, Cargo Ground Transportation, Casinos and Gaming, Coal and Consumable Fuels, Commercial and Residential Mortgage Finance, Commercial Printing, Commodity Chemicals, Communications Equipment, Computer and Electronics Retail, Construction and Engineering, Construction Machinery and Heavy Transportation Equipment, Construction Materials, Consumer Electronics, Consumer Finance, Consumer Staples Merchandise Retail, Copper, Data Processing and Outsourced Services, Distillers and Vintners, Distributors, Diversified Capital Markets, Diversified Chemicals, Diversified Financial Services, Diversified Metals and Mining, Diversified Real Estate Activities, Diversified REITs, Diversified Support Services, Drug Retail, Education Services, Electric Utilities, Electrical Components and Equipment, Electronic Components, Electronic Equipment and Instruments, Electronic Manufacturing Services, Environmental and Facilities Services, Fertilizers and Agricultural Chemicals, Financial Exchanges and Data, Food Distributors, Food Retail, Footwear, Forest Products, Gas Utilities, Gold, Health Care Distributors, Health Care Equipment, Health Care Facilities, Health Care Services, Health Care Supplies, Health Care Technology, Heavy Electrical Equipment, Highways and Railtracks, Home Furnishings, Home Improvement Retail, Homebuilding, Homefurnishing Retail, Hotels, Resorts and Cruise Lines, Household Appliances, Household Products, Housewares and Specialties, Human Resource and Employment Services, Independent Power Producers and Energy Traders, Industrial Conglomerates, Industrial Gases, Industrial Machinery and Supplies and Components, Industrial REITs, Insurance Brokers, Integrated Oil and Gas, Integrated Telecommunication Services, Interactive Home Entertainment, Interactive Media and Services, Internet Services and Infrastructure, Investment Banking and Brokerage, IT Consulting and Other Services, Leisure Facilities, Leisure Products, Life and Health Insurance, Life Sciences Tools and Services, Marine Ports and Services, Marine Transportation, Metal, Glass and Plastic Containers, Motorcycle Manufacturers, Movies and Entertainment, Multi-line Insurance, Multi-Sector Holdings, Multi-Utilities, Office REITs, Office Services and Supplies, Oil and Gas Drilling, Oil and Gas Equipment and Services, Oil and Gas Exploration and Production, Oil and Gas Refining and Marketing, Oil and Gas Storage and Transportation, Other Specialty Retail, Packaged Foods and Meats, Paper and Plastic Packaging Products and Materials, Paper Products, Passenger Airlines, Passenger Ground Transportation, Personal Care Products, Pharmaceuticals, Precious Metals and Minerals, Property and Casualty Insurance, Publishing, Rail Transportation, Real Estate Development, Real Estate Operating Companies, Real Estate Services, Banks, Reinsurance, Renewable Electricity, Research and Consulting Services, Restaurants, Retail REITs, Security and Alarm Services, Semiconductor Materials and Equipment, Semiconductors, Silver, Soft Drinks and Non-alcoholic Beverages, Specialized Consumer Services, Specialized Finance, Specialty Chemicals, Steel, Systems Software, Technology Distributors, Technology Hardware, Storage and Peripherals, Textiles, Tires and Rubber, Tobacco, Trading Companies and Distributors, Transaction and Payment Processing Services, Water Utilities, Wireless Telecommunication Services. For the Country columns, we have the following category: Australia, Austria, Bahamas, Bangladesh, Belgium, Belize, Bermuda, British Virgin Islands, Bulgaria, Canada, Cayman Islands, China, Croatia, Curaçao, Cyprus, Czech Republic, Denmark, Estonia, Falkland Islands, Finland, France, Germany, Gibraltar, Greece, Guernsey, Hong Kong, Hungary, Iceland, India, Indonesia, Iran, Ireland, Isle of Man, Israel, Italy, Ivory Coast, Japan, Jersey, Kazakhstan, Latvia, Liberia, Liechtenstein, Lithuania, Luxembourg, Macedonia, Malaysia, Malta, Marshall Islands, Mauritius, Mexico, Monaco, Mongolia, Netherlands, Netherlands Antilles, New Zealand, Norway, Pakistan, Panama, Papua New Guinea, Philippines, Poland, Portugal, Romania, Russia, Serbia, Singapore, Slovakia, Slovenia, South Africa, South Korea, Spain, Sri Lanka, Sweden, Switzerland, Taiwan, Thailand, Ukraine, United Kingdom, United States, Uruguay, Vietnam. From this point onwards, only reply with an SQL command. You have to take all user content and the last assistant content into account. First answer should always have 'SELECT Name, Primary_Industry, Country' and 'WHERE Detailed_Description OR Primary_Industry' "},]

df = pd.read_csv('table_g.csv',encoding = "ISO-8859-1")


# React to user input
if prompt := st.chat_input("Top 10 banks in Vietnam by Revenue"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
   
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = st.session_state.messages
        )
    response = response.choices[0].message.content.strip()
    q1 = response
    if "SELECT" in response:
        response = "SELECT" + response.split("SELECT")[1]
        sql_select_list = []
        for i in df.columns:
            if i in response:
                sql_select_list.append(i)
        
        sql_select = ', '.join(sql_select_list)
        response = response.replace(response[response.find("SELECT")+7:response.find("FROM")-1], sql_select)
        
        st.session_state.messages.append({"role":"assistant","content": response})
        response
        df2 = ps.sqldf(response, locals())
        if len(df2) == 0:
            with st.chat_message("assistant"):
                st.markdown("recalibrating...")
            st.session_state.messages.append({"role": "user", "content": "You also need to check Primary_Industry OR Detailed_Description. Make sure to include 'SELECT Name, Primary_Industry, Country, Description' in your answer."})
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = st.session_state.messages
                )
            response = response.choices[0].message.content.strip()
            #response
            st.session_state.messages.append({"role":"assistant","content": response})
            sql_select_list = []
            for i in df.columns:
                if i in response:
                    sql_select_list.append(i)
            sql_select = ', '.join(sql_select_list)
            response
            df3 = ps.sqldf(response, locals())
            with st.chat_message("assistant"):
                #st.markdown(response)
                
                st.multiselect("Selected columns", df.columns, sql_select_list)
                if "Country" in sql_select_list:
                    left, right = st.columns((1, 20))
                    left.write("↳")
                    pattern = r"Country\s*=\s*'([^']+)'" 
                    matches = re.findall(pattern, response)
                    right.multiselect("Country",df["Country"].unique(), matches)
                    #abc = df[df["Country"].isin(matches)]
                if "Primary_Industry" in sql_select_list:
                    left, right = st.columns((1, 20))
                    left.write("↳")
                    pattern = r"Primary_Industry\s*=\s*'([^']+)'" 
                    matches = re.findall(pattern, response)
                    rep = []
                    for u in matches:
                        if u in df["Primary_Industry"].unique():
                            rep.append(u)
                    right.multiselect("Primary_Industry",df["Primary_Industry"].unique(), rep)
                    
                    #abc = abc[abc["Primary_Industry"].isin(rep)]
                    
                if "Detailed_Description" in sql_select_list:
                    left, right = st.columns((1, 20))
                    left.write("↳")
                    pattern = r"Detailed_Description\s+LIKE\s+'([^']*)'" 
                    matches = re.findall(pattern, response)
                    matches2 = [item.replace('%', '') for item in matches]
                    
                    right.multiselect("Detailed_Description",matches2, matches2)
                    #abc = abc[abc["Detailed_Description"].astype(str).str.contains(matches2[0])]
                df3
        
                #abc[abc.columns.intersection(sql_select_list)]
                st.download_button("Export",data=pd.DataFrame.to_csv(df3,index=False), mime='text/csv')
        else:
            with st.chat_message("assistant"):
                #st.markdown(response)
                
                #sql_select = response[response.find("SELECT")+7:response.find("FROM")-1].split(", ")
                #sql_where = response[response.find("SELECT")+7:response.find("FROM")-1].split(", ")
                st.multiselect("Selected columns", df.columns, sql_select_list)
                if "Country" in sql_select_list:
                    left, right = st.columns((1, 20))
                    left.write("↳")
                    pattern = r"Country\s*=\s*'([^']+)'" 
                    matches = re.findall(pattern, response)
                    right.multiselect("Country",df["Country"].unique(), matches)
                    #abc = df[df["Country"].isin(matches)]
                if "Primary_Industry" in sql_select_list:
                    left, right = st.columns((1, 20))
                    left.write("↳")
                    pattern = r"Primary_Industry\s*=\s*'([^']+)'" 
                    matches = re.findall(pattern, response)
                    rep = []
                    for u in matches:
                        if u in df["Primary_Industry"].unique():
                            rep.append(u)
                    right.multiselect("Primary_Industry",df["Primary_Industry"].unique(), rep)
                    #abc = abc[abc["Primary_Industry"].isin(rep)]
                    
                if "Detailed_Description" in sql_select_list:
                    left, right = st.columns((1, 20))
                    left.write("↳")
                    pattern = r"Detailed_Description\s+LIKE\s+'([^']*)'"
                    matches = re.findall(pattern, response)

                    matches2 = [item.replace('%', '') for item in matches]

                    right.multiselect("Detailed_Description",matches2, matches2)
                    #abc = abc[abc["Detailed_Description"].astype(str).str.contains(matches2[0])]
                for column in sql_select_list:
                    if is_numeric_dtype(df2[column]):
                        left, right = st.columns((1, 20))
                        _min = float(df2[column].min())
                        _max = float(df2[column].max())
                        step = (_max - _min) / 100
                        right.slider(
                            f"Values for {column}",
                            min_value=_min,
                            max_value=_max,
                            value=(_min, _max),
                            step=step,
                        )
                
                #abc[abc.columns.intersection(sql_select_list)]
                df2
                st.download_button("Export",data=pd.DataFrame.to_csv(df2,index=False), mime='text/csv')
                
    else:
        with st.chat_message("assistant"):
            st.markdown(response)
            st.markdown("sorry, error, try again :(")

