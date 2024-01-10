import streamlit as st

import langchain_helper
from html_elements import footer, website_title, website_desc, pickup_tooltip, namepage_desc


def name_page():
    # st.markdown(website_title, unsafe_allow_html=True)
    st.markdown(namepage_desc, unsafe_allow_html=True)

    options = ["Choose an option", "Shiv", "Krishna", "Ram", "Parvati", "Sita", "Lakshmi", "Ganesh", "Hanuman", "Vishnu", "Durga"]
    default_option = options[0]  # Set the default option as 'None'

    selected_option = st.selectbox("Deity Selection:", options=options, help=pickup_tooltip, index=0)
    if selected_option != 'Choose an option':
        lord_name = selected_option
        response = langchain_helper.generate_baby_names(lord_name)
        st.write("***Here are a few baby names curated by AI for your consideration.***")
        baby_names = response.strip().split(",")
        st.write("**Baby Names**")

        num_columns = 3  # Number of columns for responsiveness
        chunk_size = len(baby_names) // num_columns if len(baby_names) % num_columns == 0 else len(
            baby_names) // num_columns + 1

        cols = st.columns(num_columns)

        for i in range(num_columns):
            with cols[i]:
                sub_name_lst = baby_names[i * chunk_size: (i + 1) * chunk_size]
                for name in sub_name_lst:
                        st.write("+", name)



        # for name in baby_names:
        #     st.write("-", name)
    else:
        st.write('Kindly select an option from the dropdown menu.')


    st.markdown(footer, unsafe_allow_html=True)