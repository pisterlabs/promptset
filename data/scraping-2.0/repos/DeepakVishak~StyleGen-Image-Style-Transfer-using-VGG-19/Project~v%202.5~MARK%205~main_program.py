import streamlit as st
from homepage.MainPage import Home,WhatWeDo,HowToUse,Contact,Footer
from stylegen.main import Main
import openai
import os

class MainProgram:

    def __init__(self):
        pass

    # Create a function to show the homepage
    def show_homepage(self):
        h = Home()
        wd = WhatWeDo()
        htu = HowToUse()
        c = Contact()
        f = Footer()
        h.home_display()
        wd.whatwedo_display()
        htu.howtouse_display()
        c.contact_display()
        f.footer_display()

    # Create a function to show the StyleGen page
    def show_stylegen(self):
        openai.api_key = os.getenv("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")
        main = Main()
        main.main_display()

    def mainprogram_display(self):


        # Define the options for the side menu
        options = ["🏠 Homepage", "🖌️ Try StyleGen"]

        # Set the default option to "Homepage"
        selected_option = options.index("🏠 Homepage")

        # Create the side menu
        selected_option = st.sidebar.selectbox("Select an option", options)


        # Show the appropriate page based on the selected option
        if selected_option == "🏠 Homepage":
            self.show_homepage()
        elif selected_option == "🖌️ Try StyleGen":
            self.show_stylegen()




mp = MainProgram()
st.set_page_config(page_title="StyleGen",layout="wide",page_icon="🖌️")
mp.mainprogram_display()