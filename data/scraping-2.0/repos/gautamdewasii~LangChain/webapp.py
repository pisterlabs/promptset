import streamlit as st
import langchain_helper


st.title("COMPUTER SHOP NAME GENERATOR")

country_name=st.sidebar.selectbox("select any country",("India","USA","UAE","South Korea"))



if country_name:
    response=langchain_helper.generate_shop_name_and_services(country_name)
    st.header(response['shop_name'])
    services=response['services'].split(",")
    st.write("SERVICES ARE :- ")
    for service in services:
        st.write(service)
