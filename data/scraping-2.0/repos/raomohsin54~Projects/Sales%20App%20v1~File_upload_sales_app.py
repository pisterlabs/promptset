import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# Set page configuration
st.set_page_config(
    page_title="Sales Analysis App",
    page_icon=":sales:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define styles for the app
styles = """
<style>
img {
    max-width: 50%;
}
.sidebar .sidebar-content {
    background-color: #f5f5f5;
}
</style>
"""

# Render styles
st.markdown(styles, unsafe_allow_html=True)

image = Image.open(r"C:\Users\mmukhtiar\Downloads\20-Easy-Call-Center-Sales-Tips-to-Increase-Sales-1024x536.png")
image2 = Image.open(r"C:\Users\mmukhtiar\Downloads\sales-prediction.jpg")

# Define header
header = st.container()
with header:
    st.image(image)
    st.title("Sales Analysis App")
    st.markdown("Use this Streamlit app to analyze your sales!")
    st.write("")

# Define main content
content = st.container()
with content:
    # Load sales dataset
    sale_file = st.file_uploader('Select Your Local Sales CSV (default provided)')
    if sale_file is not None:
        sale_df = pd.read_csv(sale_file, encoding='latin-1')
    else:
        st.warning("Please select a CSV file to continue.")
        st.stop()

    # Select x and y variables
    st.subheader("Create a scatterplot")
    st.write("Select the x and y variables to create a scatterplot.")
    col1, col2 = st.beta_columns(2)
    with col1:
        selected_x_var = st.selectbox('X variable', ['QUANTITYORDERED', 'PRICEEACH', 'SALES'])
    with col2:
        selected_y_var = st.selectbox('Y variable', ['PRICEEACH', 'QUANTITYORDERED', 'SALES'])

    # Create scatterplot
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x = sale_df[selected_x_var], y = sale_df[selected_y_var], hue = sale_df['PRODUCTLINE'])
    plt.xlabel(selected_x_var)
    plt.ylabel(selected_y_var)
    plt.title("Scatterplot of Sales")
    st.pyplot(fig)

# Define sidebar
sidebar = st.sidebar
with sidebar:
    st.image(image2)
    st.subheader("Get insights about the data")
    st.write("Enter a prompt to generate insights about the data using PandasAI and OpenAI.")
    prompt = st.text_input("Enter your prompt:")
    if prompt:
        # Initialize PandasAI and OpenAI
        llm = OpenAI()
        pandas_ai = PandasAI(llm)
        
        # Run PandasAI with user input prompt
        result = pandas_ai.run(sale_df, prompt=prompt)
        
        # Display result
        if result is not None:
            st.write("### Insights")
            st.write(result)
