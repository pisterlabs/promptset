from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from PIL import Image
import matplotlib.pyplot as plt
# Configure Matplotlib to use the TkAgg backend
st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable warning
plt.switch_backend('TkAgg')  # Set the backend
load_dotenv()
API_KEY = os.environ['OPEN_API_KEY']

llm = OpenAI(api_token=API_KEY)
pandas_ai = PandasAI(llm, conversational=False)  

# Main app
def main():
    st.sidebar.title("MagicAnalyser")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        st.session_state.uploaded_data = pd.read_csv(uploaded_file)  # Store data in session state
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Go to", ("Analysis", "User Manual"))
    if page == "Analysis":
        analysis_page()
    elif page == "User Manual":
        usage_page()


def analysis_page():    
    if hasattr(st.session_state, 'uploaded_data'):
        st.write("Uploaded file content:")
        st.dataframe(st.session_state.uploaded_data)
        st.write("Column names:")
        column_names = list(st.session_state.uploaded_data.columns)
        st.write(column_names)         
        question = st.text_area("Enter your analysis question:")            
        if st.button('submit'):
            if question:                
                with st.spinner('generating response...'):
                    answer = pandas_ai(st.session_state.uploaded_data, question)
                    st.write(answer)        
            else:
                st.warning('please ask a question')
    else:
        image = Image.open("upload.jpg")
        # Display the image
        st.image(image, caption="upload file on menu panel", use_column_width=True)        

def usage_page():
    st.header("Welcome to MagicAnalyser!")
    image = Image.open("home.jpg")
    st.image(image, caption="upload file on menu panel", use_column_width=True)
    st.write("MagicAnalyser is a data analysis tool that helps you gain insights from your data. Follow these steps to use the application:")

    st.subheader("Step 1: Upload Data")
    st.write("1. Start on the 'Home' page.")
    st.write("2. Upload your data file (CSV or Excel) using the 'Upload a CSV or Excel file' section on the left panel.")
    st.write("3. Once the file is uploaded, it will be stored in the application.")

    st.subheader("Step 2: Perform Analysis")
    st.write("1. Go to the 'Analysis' page using the left panel.")
    st.write("2. View the column names of the uploaded data.")
    st.write("3. Enter an analysis question in the text input box under 'Enter your analysis question:'.")

    st.subheader("Step 3: Submit Analysis")
    st.write("1. Click the 'Submit' button to perform the analysis.")
    st.write("2. The application will provide insights or visualizations based on your question.")

    st.subheader("Step 4: Explore Data")
    st.write("1. Explore the uploaded data and the results of your analysis.")
    st.write("2. View the uploaded data's content, number of rows, and file size.")

    st.subheader("Additional Information")
    st.write("1. You can navigate between different pages using the left panel.")
    st.write("2. The 'Home' page displays an image, introductory content, and sample charts.")
    st.write("3. The 'Usage' page provides this user manual.")

    st.write("Enjoy using MagicAnalyser to analyze your data and discover valuable insights!")

    pass

if __name__ == "__main__":
    main()
