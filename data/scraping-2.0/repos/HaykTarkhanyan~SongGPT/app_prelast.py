import streamlit as st
import os
import openai
from PIL import Image
from datetime import datetime
import time


openai.api_key = os.environ['OPENAI_API_KEY']


def get_text_from_company_info(company_name=None, industry=None, sector=None, description=None):
    company_name_text = ""
    industry_text = ""
    description_text = ""
    sector_text = ""
    if company_name:
        company_name_text = f"Name of the company is {company_name}. "
    if industry:
        industry_text = f"Company specializes in {industry} industries."
    if sector:
        sector_text = f"Company operates in {sector} sector. "
    if description:
        description_text = description
    
    return company_name_text + industry_text + sector_text + description_text



def main():
    st.set_page_config(page_title="Text Generation Prompt Engineering", 
                       page_icon=":robot_face:", layout="wide")
    st.title("Text Generation Prompt Engineering")
    
    purpose = st.selectbox("Purpose", ["mission", "vision", "general_overview", "main_phases", \
        "competitive_advantage", "SWOT", "roadmap", "current_status", "marketing_strategy", "Another option..."]) 
    # Create text input for user entry
    if purpose == "Another option...": 
        purpose = st.text_input("Enter your other option here")
    
    st.write("Below is the field for the initial instruction for GPT. We are just preparing it for input. This text is crucial to get right")
    prompt = st.text_input("Prompt")
    
    st.write("Based on the inputs below we will generate the request to GPT, you can skip some of the inputs. The request is generated in the following format:")
    st.write("Name of the company is {company_name}. Company specializes in {industry}. Company operates in {sector} sector. {description}")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.write("Below are the inputs for the request to GPT, you can skip all of them, except one.")
    company_name = st.text_input("Company Name")
    industry = st.text_input("Industry (write as many as you want, separated by comma)")
    sector = st.text_input("Sector (write as many as you want, separated by comma)")
    description = st.text_input("Description")
    model = st.selectbox("ChatGPT version (gpt 4 is newer but slower)", ["gpt-3.5-turbo", "gpt-4"])
    # file_name = st.text_input("File name (you can download everything to a file in the end)", value=f"{purpose}_{date_time}.txt")

    if st.button("Generate"):
        if not any([company_name, industry, sector, description]):
            st.warning("Please provide at least one input (company_name, industry, sector, description)")
            return
        if prompt is None or prompt == "":
            st.warning("Please provide a prompt")
            return
        
        # Start time
        start_time = time.time()
        
        # generate the request to GPT
        request = get_text_from_company_info(company_name=company_name, industry=industry, sector=sector, description=description)
        messages = [{"role": "system", "content": prompt}, 
                    {"role": "user", "content": request}]
        st.write("### Request to GPT:")
        st.write(request)
            
        # generate the response from GPT
        response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages)
        
        response_text = response['choices'][0]['message']['content'].strip()
        
        # End time
        end_time = time.time()
        
        # Calculate duration
        duration = end_time - start_time
        
        # Display the duration
        st.write("### Time taken to generate the response:")
        st.write(f"{duration:.2f} seconds")
        
        # display the response
        st.write("### Response from GPT:")
        st.write(response_text)
        
        # text to save
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        
        to_save = f"{date_time}\n{purpose} slide\nlanguage model: {model}\nTime taken: {duration:.2f} seconds\n\nPrompt used:\n {prompt}\n\nInput:\n {request}\n\nResponse:\n {response_text}\n\n"
        print(to_save)
        # download the file  
        st.markdown("### Download")  
        st.write("Note: in the filename the time may not be in your timezone")
        st.write("**Note:** pressing download button will refresh the page and youll lose everything under `Generate` button")
        file_name = f"{purpose}_{date_time}.txt"    
        if st.download_button(label="Download everything to a file", data=to_save, file_name=file_name, mime="text/plain"):
            st.write("Downloaded")
    
    # menu = ["Generation", "Generation Detailed"]
    # choice = st.sidebar.selectbox("Select an option", menu)

    # if choice == "Generation":
    
    # if st.download_button(label="Download", data=images[0]['url'], file_name=f"{prompt}.png", mime="image/png"):
    #     st.write("Downloaded")

if __name__ == "__main__":
    main()
