import streamlit as st
import openai
import pandas as pd
import time

# Set the OpenAI API Key (Replace 'XXXXXXXXXXXXX' with your actual API key)
openai.api_key = 'XXXXXXXXXXXXX'

# Custom CSS to style the app
# Create a Streamlit app with improved design
st.set_page_config(page_title="AI-Powered Business Idea Generator", layout="wide")

st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_business_idea(country, countryOfResidence):
    with st.spinner("AI is thinking..."):
        time.sleep(3)

    prompt = f"Generate a business idea for people from {country} living in {countryOfResidence}, please provide three examples as to why they are great business ideas."
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=100
    )

    idea = response.choices[0].text.strip()
    return idea

# Load the dataset
file_path = 'Datasets/Permanent Resident - Country of Citizenship.xlsx'
df = pd.read_excel(file_path)

# Clean the data
df = df.dropna().reset_index(drop=True)
df = df[['Unnamed: 1', 'Unnamed: 3', 'Unnamed: 4']]
df.columns = ['Country', 'Percentage', 'Base Count']
df = df.iloc[1:]
df['Percentage'] = pd.to_numeric(df['Percentage'], errors='coerce')

# Exclude the total count row and get the specific country with the max percentage
specific_country_df = df.iloc[1:]
specific_country_max_percentage = specific_country_df.loc[specific_country_df['Percentage'].idxmax()]



# Header
st.title("AI-Powered Business Idea Generator")
st.write("Welcome to the AI-powered Business Idea Generator. Our AI model has analyzed data from various countries and industries to provide you with unique business ideas.")
st.write("Simply select a country, and we'll use the power of AI to suggest business ideas tailored to people from that country living in your chosen location.")

# Sidebar for user input
st.sidebar.header("Select Options")
country = st.sidebar.selectbox("Select Country", df['Country'])
countryOfResidence = st.sidebar.text_input("Enter Country of Residence", "Canada")

# Main content area with accordion and radio buttons
option = st.radio('Choose an option:', ('Show Data Info', 'Generate Business Idea'))
if option == 'Show Data Info':
    with st.expander('Data Information'):
        st.write("Country with max percentage:", specific_country_max_percentage['Country'])
        st.write("Percentage:", specific_country_max_percentage['Percentage'])
        st.write("Base Count:", specific_country_max_percentage['Base Count'])
elif option == 'Generate Business Idea':
    with st.expander('Business Idea'):
        # This space will be updated with the business idea when the "Generate Business Idea" button is pressed
        if st.button("Generate Business Idea"):
            idea = get_business_idea(country, countryOfResidence)
            st.write("Business idea for people from", country, "living in", countryOfResidence, ":")
            st.write(idea)