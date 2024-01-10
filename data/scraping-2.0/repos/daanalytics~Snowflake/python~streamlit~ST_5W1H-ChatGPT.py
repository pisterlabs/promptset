import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
import openai
#import json
import re #regular expressions

# Set AI Model Engine & OpenAI API Key
ai_model_engine = 'gpt-3.5-turbo'
load_dotenv('openai_env.env')
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Set Page Config
st.set_page_config(layout="wide")
st.title('Chatting with ChatGPT using the 5W1H Method')

# Set Side-bar
st.sidebar.image("https://github.com/daanalytics/Snowflake/blob/master/pictures/5W1H_ChatGPT_Blackbox.png?raw=true", use_column_width=True)
st.sidebar.title('5W1H Method')

st.sidebar.markdown(
"""
- **Who?**
    - Understanding "who" can help in tailoring the language and complexity of the response.
- **What?**
    - Specifying "what" ensures that the AI provides the type and format of information you desire.
- **Where?**
    - Defining "where" can help in receiving region or context-specific answers.
- **Why?**
    - Knowing "why" can help in determining the depth and angle of the AI's response.
- **When?**
    - Framing "when" can help narrow down the context of the information provided.
- **How?**
    - Clarifying "how" can guide the AI in structuring its answer in the most useful way.            
"""
)

def main():

    # 1. Collecting user input according to the 5W1H Method
    # 2. Sending user input to ChatGPT function
    # 3. Display the ChatGPT response
    
    # Variable Who --> audience?
    who = st.text_area('Who', help='Are you aiming the prompt for a software developer, a student, a researcher, or a layperson?')

    # Variable What --> what should be done?
    what = st.text_area('What', help='Are you looking for a detailed explanation, a summary, code, or maybe a list??')

    # Variable Where --> where will the output be used?
    where = st.text_area('Where', help='Are you asking about a concept''s application in a specific country, industry, or environment?')

    # Variable Why --> what's the goal?
    why = st.text_area('Why', help='Are you asking because you want a deep understanding, or are you trying to compare concepts?')

    # Variable When --> when will 'it' happen?
    when = st.text_area('When', help='Are you asking about historical events, contemporary issues, or future predictions?')

    # Variable How --> style, structure, length, use of language, etc.
    how = st.text_area('How', help='Are you seeking a step-by-step guide, an overview, or perhaps a methodological explanation?')

    prompt_output = who + ' ' + what + ' ' + where + ' ' + why + ' ' + when + ' ' + how

    # Submit Button

    form = st.form(key='5W1H_form')

    submit = form.form_submit_button(label='Submit')

    if submit:
        # Submit the prompt output to ChatGPT
        openai_response = get_openai_response(prompt_output)

        # Regular expression to extract topics and descriptions
        pattern = r'\d+\.\s(.*?):\s(.*?)(?=\d+\.|$)'
        topics_and_descriptions = re.findall(pattern, openai_response, re.S)
        topic_num = 0

        for topic, description in topics_and_descriptions:  

            topic_num = topic_num + 1

            st.markdown('**'+str(topic_num)+'**' + ': ' + topic + ' - ' + description)    

        #st.json(openai_reponse)


def get_openai_response(prompt_output):

    # Use this function to generate a ChatGPT response
    # 1. Submit the prompt_output to the OpenAI API
    # 2. Return the chat response

    # This endpoint might change depending on OpenAI's updates
    endpoint = "https://api.openai.com/v1/chat/completions"  
    headers = {
        "Authorization": f"Bearer  {openai.api_key}",
        "Content-Type": "application/json",
    }
    
    chat_completion = openai.ChatCompletion.create(
        model=ai_model_engine,
        messages=[
            {"role": "system", "content": "You are a community manager, skilled in writing informative blogposts about the subject area of Data & AI in general and the Snowflake Data Cloud in specific."},
            {"role": "user", "content": prompt_output}
        ]
    )
    
    #openai_reponse = json.loads(chat_completion['choices'][0]['message']['content'])
    openai_response = chat_completion['choices'][0]['message']['content']

    #try:
    #    json_content = json.loads(openai_reponse)
    #except json.JSONDecodeError:
    #    print(f"Failed to decode JSON from content: {openai_reponse}")
    #    # Handle error, e.g., set json_content to a default value or take some corrective action
    #    json_content = {}

    return openai_response

# Execute the main function
main() 