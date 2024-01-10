import openai
import streamlit as st
import json
import base64
import time

# Set OpenAI API key
openai.api_key = st.secrets["pass"]

# About Me
st.sidebar.title("About Me")
st.sidebar.markdown("Welcome to Code generation!\n\n"
                    "I am an OpenAI-powered script generator."
                    " With just a few inputs from you, I can create Python scripts that generate insightful reports and visualizations.")

# How to Use
st.sidebar.title("How to Use")
st.sidebar.markdown("1. Select the graph type from the right panel dropdown.\n"
                    "2. Select the data source from the radio button list.\n"
                    "3. Enter the query to generate a Python script.\n"
                    "4. Click the Generate button.\n"
                    "5. Copy the script and use it in your Python environment to generate reports and statistical charts.")

# Insert and scale down the GIF at Column 1 and Text at Column 2

# Create two columns: one for the GIF and one for the header
col1, col2 = st.columns([1, 3])

# In the first column, display the GIF
gif_path = 'C:/Users/Fezde/PycharmProjects/OpenAI_GenAI/Data/HomePageBOT.gif'  # Replace with the path to your GIF file
width = 160  # Set the desired width for scaling down the GIF
height = 160  # Set the desired height for scaling down the GIF

# Generate HTML for displaying the scaled-down GIF
html_code = f'<img src="data:image/gif;base64,{base64.b64encode(open(gif_path, "rb").read()).decode()}" width="{width}" height="{height}" />'

# Display the scaled-down GIF using Markdown and HTML
with col1:
    st.markdown(html_code, unsafe_allow_html=True)

# In the second column, display the header with changed font color
with col2:
    st.markdown('<h1 style="color: blue; margin-bottom: 30px; font-size: 52px;">Code Generation Using Generative AI</h1>', unsafe_allow_html=True)


# Options to select type of Action
options = ['Pie Chart', 'Histogram', 'Bar Chart', 'Line Chart', 'Scatter Plot', 'Box Plot', 'Heat Map', 'Best Fit']
st.subheader("Select Graph Type")
selected_option = st.selectbox('Please select a Graph Type from dropdown', options)

# Data source selection
datasource_options = ['SQL', 'Snowflake', 'Excel', 'CSV']
st.subheader("Select Data Source")
data_source = st.selectbox('Please select a data source from dropdown', datasource_options)

# Prompt input
st.subheader("Enter Query to Generate Code")
user_query = st.text_area("User query (mandatory)",
                          help="Enter the requirements to generate code in Python for the selected graph and data source.")

# Generate query output
if st.button("Generate"):
    # Setting the prompt
    prompt = f"Give the user a script that would generate a {selected_option} graph for the following " \
             f"user query, considering the data source is {data_source}:\n" \
             f"User query: {user_query}\n"

    # Create an empty placeholder for the progress bar and percentage text
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)  # Initialize the progress bar

    with st.spinner("Generating python script for requested query..."):
        st.snow()

        for percent_complete in range(1, 82):
            time.sleep(0.02)  # Simulate progress delay
            progress_bar.progress(percent_complete)  # Update progress bar


        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system",
                 "content": "You are an assistant that helps to create Python Script. "
                            "Consider the following techniques to show the result:\n\n"
                            "1. Share an optimized and effective result for the requested query.\n"
                            "2. The code should have proper comments and be easy to read.\n"
                            "3. Share detailed instructions with procedure numbers on how to run the code.\n"
                            "4. Help with detailed information about how to connect or fetch data from the data source.\n"
                            "5. Make sure to share a sequential Python script in PEP8 format.\n"
                            "If you receive a task that does not seem relevant to your role, "
                            "reply with a joke and suggest how the user can use you."
                 },
                {"role": "user",
                 "content": prompt}
            ]
        )

        # Show spinner with balloons animation
        st.snow()

        # Simulate progress while waiting for API response
        for percent_complete in range(83, 101):
            time.sleep(0.02)  # Simulate progress delay
            progress_bar.progress(percent_complete)  # Update progress bar

        # Remove the progress bar
        progress_bar.empty()

        # Retrieve the generated text from API response
        generated_text = response.choices[0].message.content

    # Display the generated text
    st.subheader("Output")
    st.code(generated_text)

#python -m streamlit run main.py