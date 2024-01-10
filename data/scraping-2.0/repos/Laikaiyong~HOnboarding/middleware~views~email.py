import streamlit as st
import os
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate

#################################################################################################################################

prompts = {} #pre-defined prompt templatess, include "{user_input}" to merge input content
inputs = {} #used to merge into prompt templates, merged into the "{user_input}" placeholder
defaults = {} #used for default values in simple examples

#################################################################################################################################
# PROMPTS
#################################################################################################################################
prompts[""] = """
{user_input}

According to above respond with the following actions in sections.

Title: Section 1 - Reply Template
Action: Please write a reply to the above text:

Title: Section 2 - Summarize
Action: Please summarize the above content:

Title: Section 3 - Sentiment
Action: Sentiment of the above content (Positive or negative):

Title: Section 4 - Action Recommendation
Action: Recommended next step based on the above content:
"""

#################################################################################################################################

prompts["Reply Template"] = """
{user_input}

Please write a reply to the above text:
"""

#################################################################################################################################

prompts["Summarize"] = """
{user_input}

Please summarize the above content:
"""

#################################################################################################################################

prompts["Sentiment"] = """
{user_input}

Sentiment of the above content (Positive or negative):
"""

#################################################################################################################################

prompts["Recommendation"] = """
{user_input}

Recommended next step based on the above content:
"""

#################################################################################################################################
# INPUTS
#################################################################################################################################

inputs["Social Media"] = """
Dear Hong Leong Bank,

I am writing to express my dissatisfaction with the services I have recently experienced with Hong Leong Bank. I believe it is crucial to bring the following matters to your attention for necessary investigation and resolution. 
I found that there was a lack of clarity in the communication provided, especially regarding fees, terms, and conditions. This has resulted in confusion and frustration on my part.

Sincerely,
Julia Shong
"""

#################################################################################################################################

inputs["Complimentary Customer Email"] = """
Dear Hong Leong Bank,

I am writing to compliment one of your company representatives, Ms Sara. I recently had the pleasure of speaking with Sara regarding my personal loan request. Sara was extremely helpful and knowledgeable, and went above and beyond to ensure that all of my questions were answered.
She helped me to gain a better understanding of the loan eligibility, documents required, application and repayment process. I would be happy to recommend Hong Leong Bank to others based on my experience.

Sincerely,
Jacky Wong
"""

#################################################################################################################################

inputs["Complaint Email"] = """
Dear Hong Leong Bank Customer Service,

I am writing to express my concern and frustration regarding a persistent issue with my account that has yet to be resolved.
On 1/11/2023, I first noticed a discrepancy in my account statement, specifically related to incorrect charges and missing deposits.
I believe there might be an error in my account and I would like to clarify with your side. 
Despite my attempts to resolve this matter through your customer service hotline, the issue remains unresolved, and it has been causing significant inconvenience and distress. 

I hope representatives on your side can help to resolve this issue soon.

Yours sincerely,
Jacky Wong
"""

def get_llm():
    
    model_kwargs = { #AI21
        "maxTokens": 1024, 
        "temperature": 0, 
        "topP": 0.5, 
        "stopSequences": [], 
        "countPenalty": {"scale": 0 }, 
        "presencePenalty": {"scale": 0 }, 
        "frequencyPenalty": {"scale": 0 } 
    }
    
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        model_id="ai21.j2-ultra-v1", #set the foundation model
        model_kwargs=model_kwargs) #configure the properties for Claude
    
    return llm


def get_prompt(user_input, template):
    
    prompt_template = PromptTemplate.from_template(template) #this will automatically identify the input variables for the template

    prompt = prompt_template.format(user_input=user_input)
    
    return prompt


def get_text_response(user_input, template): #text-to-text client function
    llm = get_llm()
    
    prompt = get_prompt(user_input, template)
    
    return llm.predict(prompt) #return a response to the prompt




def load_view(): 
    st.markdown(
        f'''
        <style>
            .reportview-container .sidebar-content {{
                padding-top: 0rem;
            }}
            .reportview-container .main .block-container {{
                padding-top: 0rem;
                margin-top: 0rem;
            }}
        </style>
        ''',
        unsafe_allow_html=True,
    )  
    # st.set_page_config(page_title="Demo Showcase", layout="wide")

    col1, col2, col3 = st.columns(3)


    with col1:
        
        with st.expander("View prompt"):

            selected_prompt_template_text = prompts[""]

            prompt_text = st.text_area("Prompt template text:", value=selected_prompt_template_text, height=350)
        
        
    with col2:
        st.subheader("Customer Text")
        inputs_keys = list(inputs)
        
        input_selection = st.selectbox("Select an input example:", inputs_keys)
        
        selected_input_template_text = inputs[input_selection]

        input_text = st.text_area("Input text:", value=selected_input_template_text, height=350)
        
        process_button = st.button("Go", type="primary")



    with col3:
        st.subheader("Result")
        
        if process_button:
            with st.spinner("Running..."):
                response_content = get_text_response(user_input=input_text, template=prompt_text)

                st.write(response_content)
