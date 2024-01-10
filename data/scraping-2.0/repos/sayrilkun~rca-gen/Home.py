#
# Initialization
#
from distutils.command.upload import upload
import email
import sys
sys.path.append('./lib/')

#
# Imports
#
import openai
import streamlit as st
from streamlit_chat import message
from lib import docx_util
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_extras.app_logo import add_logo
import email_util
import prompts
from lib import ruthinit
from lib import filechecker
from lib import email_parser
import time
import os
import datetime
import json
#
# Globals
#
log = ruthinit.log
database = ruthinit.database
container = ruthinit.container
file = False
image = Image.open('static/ruthname.png')
is_rca_det_generated = False # flag if rca_det is generated
is_file_generated = False # flag if file is generated
full_rca_det = []

# Generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    completion = openai.ChatCompletion.create(
        model=model,
        messages=st.session_state['messages']
    )
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens

# Open AI API Function
def prompt(user_input):
    try:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
        log.info("AI responded")
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        log.info(st.session_state['generated'])
        # st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        cost = total_tokens * 0.002 / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

    except openai.error.InvalidRequestError:
        st.warning('Invalid Request. Restart app and try again')

    

# Setting page title and header
st.set_page_config(page_title="Ruth", page_icon= "static/ruthlogo.png")
#st.markdown("<h1 style='text-align: center;'> 🤖 Ruth: RCA GENERATOR🤖 </h1>", unsafe_allow_html=True)
st.markdown("""---""")

# Set org ID and API key
openai.organization = st.secrets.secrets["openai"]["organization"][0]
openai.api_key = st.secrets.secrets["openai"]["api_key"][0]



# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0
prompt_generated = False
# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
add_logo('static/ruthsmall.png', height=100)
st.sidebar.header("Main Page")
#st.sidebar.title("Team DATAMRK")
counter_placeholder = st.sidebar.empty()
# counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Set model
model_name = "GPT-3.5"
model = "gpt-3.5-turbo-16k"
# model = "gpt-4"


# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    prompt_generated = False
    # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


file_container = st.container()
file=False
#st.markdown("##")


with file_container:
    #st.chat_message("Hi! My name is Ruth, please upload your email file so I can start generating your RCA!", avatar='static/ruthlogo.png')
    message("Hi! My name is Ruth, please upload your email file so I can start generating your RCA!", key="intro", avatar_style="micah", seed = "Rascal")
    uploaded_file = st.file_uploader(" ",type=['.eml', '.msg'])
    generate_button = st.button("Generate RCA", key="generate",use_container_width=True)

# INITIAL PROMPT (incident time line)
if uploaded_file != None:
    log.info(uploaded_file)

    ext = filechecker.GetFileExtension(uploaded_file.name)
    log.info(f"extension {ext}")

    if ext == '.eml':
        file = True
        log.info("Processing eml file")
        bytes_data = uploaded_file.getvalue()
        parsed_mail = email_util.parse_from_bytes(bytes_data)

    elif ext == '.msg': #v2 of parsing .msg
        file = True
        log.info("processing msg file")
        parsed_mail = email_parser.convert_msg_to_text(uploaded_file)

    # elif ext == '.msg':
    #     file = True
    #     eml = email_parser.convert_msg_to_eml(uploaded_file)
    #     parsed_mail = email_util.parse_from_bytes(eml.as_bytes())

    # INITIAL PROMPT, OTHER PROMPTS INSIDE PROMPTS.PY
    inc_timeline_prompt = f'''Shortly summarize the contents of this email one by one thread per timestamp using only one or two sentences. Summarize the contents don't just copy it. 

    {parsed_mail}
    Can you please follow this format (No special characters no \\n or whatever):

    {prompts.inc_timeline_format}

    '''
# IF BUTTON IS CLICKED
if generate_button:
    if file is True:
        with st.spinner('Generating Incident Timeline. Please Wait...'):
            log.info("Sending Message")
            log.info(inc_timeline_prompt)
            prompt(inc_timeline_prompt)
            time.sleep(3)
        with st.spinner('Generating Root Cause Analysis. Please Wait...'):
            prompt(f"{prompts.rca_details_prompt}")
            time.sleep(3)
        with st.spinner('Generating Action Items. Please Wait...'):
            prompt(prompts.action_items_prompt)
            time.sleep(3)
        with st.spinner('Generating 5 WHYs Analysis. Please Wait...'):
            prompt(prompts.five_whys_prompt)
            prompt_generated = True
            # prompt("what is A")
            # file = False

# container for chat history
response_container = st.container()
# container for text box
rca_container = st.container()

# chat box
# with container:
#     with st.form(key='my_form', clear_on_submit=True):
#         user_input = st.text_area("You:", key='input', height=100)
#         submit_button = st.form_submit_button(label='Send')

#     if submit_button and user_input:
#         prompt(user_input)

#chat conversation
if st.session_state['generated']:
    rca_details_clean = ""
    inc_timeline_clean = ""

    with response_container:
        for i in range(len(st.session_state['generated'])):
            # HIDING THE CHATBOX
            # message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="croodles", seed="Tigger")
            # message(f'XX{i}XX {st.session_state["generated"][i]} ', key=str(i), avatar_style="bottts", seed = "Sophie")
            log.info(st.session_state['generated'])
            log.info(f"Number of tokens: {st.session_state['total_tokens'][i]};")
            try:
                # CONVERT THE RESPONSE TO DATAFRAME
                inc_timeline_clean = st.session_state["generated"][0].replace("\\n", "").replace("\\'", "")
                inc_timeline_df = pd.DataFrame(eval(inc_timeline_clean))
                
            except Exception as e:
                pass



        # SECOND PROMPT (RCA DETAILS)

        rca_root_cause = "" # Root Cause
        rca_ex_sum = "" # executive summary
        rca_inv_res = "" # investigation & resolution
        rca_cont_fact = "" # contributing factors


        st.header("RCA Details")
        # rca_details_button = st.button("Generate RCA Details", key="rca_details",use_container_width=True)
        # if rca_details_button:
            # prompt(f"{prompts.rca_details_prompt}")
        # st.write(st.session_state["generated"][1])
        fail = False

        try:
            log.info("Generating RCA Details")
            rca_details_clean = st.session_state["generated"][1].replace("\\n", "").replace("\\'", "")
            rca_details_clean = rca_details_clean.replace("'s", "\\'s")
            log.info(rca_details_clean)
            # st.write(rca_details_clean)
            rca_details_df = pd.DataFrame(eval(rca_details_clean))


            st.subheader("Root Cause")
            rca_root_cause = rca_details_df.iloc[0, 0]
            st.success(rca_root_cause)

            st.subheader("RCA Executive Summary")
            rca_ex_sum = rca_details_df.iloc[1, 1]
            st.success(rca_ex_sum)

            st.subheader("Investigation & Resolution")
            rca_inv_res = rca_details_df.iloc[2, 2]
            st.success(rca_inv_res)

            st.subheader("Contributing Factors")
            rca_cont_fact = rca_details_df.iloc[3, 3]
            st.success(rca_cont_fact)

            log.info("Generated RCA Details")

            is_rca_det_generated = True

        
        except Exception as e:
            log.info(e)
            fail = True


        st.header("Action Items")
        try:
    #         action_items="[{'Actions': 'Diagnostic', 'Description': 'Investigate potential misconfiguration in the payment gateway integration causing CPU and memory spikes', 'Owner': 'Tyrone Guevarra', 'Date': '10th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Diagnostic', 'Description': 'Further investigate the potential misconfiguration in the integration of the new payment gateway', 'Owner': 'Mary Rose Ann Guansing', 'Date': '10th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Diagnostic', 'Description': 'Explore the connection between the new payment gateway and the system issues', 'Owner': 'Johndell Kitts', 'Date': '10th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Diagnostic', 'Description': 'Investigate the intriguing behavior in the payment processing code', 'Owner': 'John Michael Dy', 'Date': '9th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Diagnostic', 'Description': 'Investigate the surge in deadlock incidents and their impact on transaction delays', 'Owner': 'Redner Cabra', 'Date': '9th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Implementation', 'Description': 'Swiftly resolve the point-of-sale system issue', 'Owner': 'Team', 'Date': '9th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Implementation', 'Description': 'Address the critical issue causing transaction failures and disruptions', 'Owner': 'Team', 'Date': '9th August 2023', 'Status': 'Not Completed'}]"
            action_items_clean = st.session_state["generated"][2].replace("\\'","").replace("\\n","")
            action_items_clean = action_items_clean.replace("'s", "\\'s").replace("'t", "\\'t")
            action_items_df = pd.DataFrame(eval(action_items_clean))
            st.table(action_items_df)
            
        except Exception as e:
            st.write("error parsing")

        st.header("RCA 5 WHYs")
        rca_whys = st.session_state["generated"][3]
        st.success(rca_whys)
        
        st.header("Incident Timeline")
        try:
            st.table(inc_timeline_df)
            
        except Exception as e:
            st.write(st.session_state["generated"][0])

        log.info(is_rca_det_generated)
        if is_rca_det_generated:
            
            full_rca_det = [rca_root_cause, rca_ex_sum, rca_inv_res, rca_cont_fact]
            log.info(full_rca_det)
            docx_util.build_word_document(eval(action_items_clean), rca_whys, full_rca_det, eval(inc_timeline_clean))
            docx_util.convert_word_to_pdf_unix("output.docx")
            log.info("Generating Word & PDF File.")
            log.info(rca_details_clean)
            with open("output.docx", "rb") as file:
                btn = st.download_button(
                    label="Download Output File (DOCX) 📜",
                    data=file,
                    file_name="output.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            with open("output.pdf", "rb") as file:
                btnpdf = st.download_button(
                    label="Download Output File (PDF) 📜",
                    data=file,
                    file_name="output.pdf",
                    mime="application/pdf"
                )

            with st.expander("Save As"):
                incident_name = st.text_input("Incident Name")
                uploader_name = st.text_input("Uploader Name")
                save_button = st.button("Save", key="save",use_container_width=True)
                if save_button:
                    item = {
                        "categoryId": "61dba35b-4f02-45c5-b648-c6badc0cbd79",
                        "incidentDate": json.dumps(datetime.date.today(), indent=4, sort_keys=True, default=str),
                        "incidentName": incident_name,
                        "projectAssignment": "SAMPLE BANK AMS",
                        "uploader": f"{uploader_name}",
                        "emailSubject" : "[Test Email] Urgent: Point-of-Sale System Issue Resolution",
                        "rcaDetails": rca_details_clean,
                        "actionItems" : st.session_state["generated"][2],
                        "rca5WHYs" : st.session_state["generated"][3],
                        "incidentTimeline" : st.session_state["generated"][0],
                        }
                    try:
                        container.create_item(item,enable_automatic_id_generation=True)
                        st.success("Saved successfully")
                    except Exception as e:
                        st.warning(e)

            # time.sleep(3)
            # prompt(prompts.action_items_prompt)  
            # st.header("☢️ Action Items")
            # st.write(st.session_state["generated"][2])
            # try:
            #     # action_items="[{'Actions': 'Diagnostic', 'Description': 'Investigate potential misconfiguration in the payment gateway integration causing CPU and memory spikes', 'Owner': 'Tyrone Guevarra', 'Date': '10th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Diagnostic', 'Description': 'Further investigate the potential misconfiguration in the integration of the new payment gateway', 'Owner': 'Mary Rose Ann Guansing', 'Date': '10th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Diagnostic', 'Description': 'Explore the connection between the new payment gateway and the system issues', 'Owner': 'Johndell Kitts', 'Date': '10th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Diagnostic', 'Description': 'Investigate the intriguing behavior in the payment processing code', 'Owner': 'John Michael Dy', 'Date': '9th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Diagnostic', 'Description': 'Investigate the surge in deadlock incidents and their impact on transaction delays', 'Owner': 'Redner Cabra', 'Date': '9th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Implementation', 'Description': 'Swiftly resolve the point-of-sale system issue', 'Owner': 'Team', 'Date': '9th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Implementation', 'Description': 'Address the critical issue causing transaction failures and disruptions', 'Owner': 'Team', 'Date': '9th August 2023', 'Status': 'Not Completed'}]"
            #     action_items_df = pd.DataFrame(eval(st.session_state["generated"][2]))
            #     st.table(action_items_df)
                
            # except Exception as e:
            #         pass




# if prompt_generated is True:
#     st.header("☢️ RCA Details")
#     rca_details_button = st.button("Generate RCA Details :rocket:", key="rca_details",use_container_width=True)
#     # if rca_details_button:
#         # prompt(prompts.rca_details_prompt)
#         # st.write(st.session_state["generated"][1])

#     try:
        
#         rca_details="[{'Root Cause': 'The root cause of the incident is a misconfiguration in the integration of the new payment gateway. This misconfiguration is putting excessive strain on the systems resources, leading to CPU and memory usage spikes and transaction failures.', 'RCA Executive Summary': 'The email thread discusses a critical issue with the point-of-sale system that is causing transaction failures and disruptions for the client. The team explores potential leads, including the misconfigured payment gateway integration, unusual behavior in the payment processing code, and database deadlocks. Mary Rose Ann Guansing suggests the misconfiguration as a possible cause and emphasizes the need for prompt resolution due to its impact on customer satisfaction and sales. Johndell Kitts suggests investigating the connection between the new payment gateway and the system issues. John Michael Dy identifies an intriguing behavior in the payment processing code and Redner Ivan P. Cabra discovers a surge in database deadlocks during the incident. Andrei Cyril F. Gimoros sends an urgent email highlighting the importance of resolving the issue swiftly.', 'Investigation and Resolution': 'Key Dates: 10-August-23 - The email thread begins with Andrei Cyril F. Gimoros thanking the team for their dedication and mentioning the potential leads to explore. 10-August-23 - Mary Rose Ann Guansing suggests the misconfiguration and provides her upcoming out of office schedule. 10-August-23 - Johndell Kitts suggests investigating the connection between the new payment gateway and the system issues. 10-August-23 - John Michael Dy identifies an intriguing behavior in the payment processing code. 10-August-23 - Redner Ivan P. Cabra discovers the surge in database deadlocks. 10-August-23 - Andrei Cyril F. Gimoros sends an urgent email emphasizing the need for prompt resolution.'}]"
#         st.write(rca_details)
#         rca_details_df = pd.DataFrame(eval(rca_details))
#         st.table(rca_details_df)

#         st.subheader("☢️ Root Cause")
#         st.success(rca_details_df.iloc[0, 0])

#         st.subheader("☢️ RCA Executive Summary")
#         st.success(rca_details_df.iloc[0, 1])

#         st.subheader("☢️ Investigation & Resolution")
#         st.success(rca_details_df.iloc[0, 2])

#         st.subheader("☢️ Contributing Factors")
    
#     except Exception as e:
#         pass

#         # file = False
    # st.header("☢️ Action Items")
    # action_items_button = st.button("Generate Action Items :rocket:", key="action_items",use_container_width=True)
    # if action_items_button:
    #     # prompt(prompts.action_items_prompt)
    #     st.success(st.session_state["generated"][2])
#     try:
# #         action_items="[{'Actions': 'Diagnostic', 'Description': 'Investigate potential misconfiguration in the payment gateway integration causing CPU and memory spikes', 'Owner': 'Tyrone Guevarra', 'Date': '10th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Diagnostic', 'Description': 'Further investigate the potential misconfiguration in the integration of the new payment gateway', 'Owner': 'Mary Rose Ann Guansing', 'Date': '10th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Diagnostic', 'Description': 'Explore the connection between the new payment gateway and the system issues', 'Owner': 'Johndell Kitts', 'Date': '10th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Diagnostic', 'Description': 'Investigate the intriguing behavior in the payment processing code', 'Owner': 'John Michael Dy', 'Date': '9th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Diagnostic', 'Description': 'Investigate the surge in deadlock incidents and their impact on transaction delays', 'Owner': 'Redner Cabra', 'Date': '9th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Implementation', 'Description': 'Swiftly resolve the point-of-sale system issue', 'Owner': 'Team', 'Date': '9th August 2023', 'Status': 'Not Completed'}, {'Actions': 'Implementation', 'Description': 'Address the critical issue causing transaction failures and disruptions', 'Owner': 'Team', 'Date': '9th August 2023', 'Status': 'Not Completed'}]"
#         action_items_df = pd.DataFrame(eval(st.session_state["generated"][2]))
#         st.table(action_items_df)
        
#     except Exception as e:
#         st.write("error parsing")

#     st.header("☢️ RCA 5 WHYs")
#     five_whys_button = st.button("Generate 5 WHYs :rocket:", key="five_whys",use_container_width=True)
#     # if five_whys_button:
#         # prompt(prompts.five_whys_prompt)
#         # st.write(st.session_state["generated"][1])
#     five_whys = "1. Why are there system hang-ups and transaction failures?\n   - Possible cause: Misconfigured payment gateway integration.\n\n2. Why is there a misconfigured payment gateway integration?\n   - Possible cause: Issues during the deployment process.\n\n3. Why were there issues during the deployment process?\n   - Possible cause: Lack of proper configuration and testing.\n\n4. Why was there a lack of proper configuration and testing?\n   - Possible cause: Insufficient attention to detail or oversight.\n\n5. Why was there insufficient attention to detail or oversight?\n   - Possible cause: Lack of clear communication or guidelines during the deployment process."
#     st.success(five_whys)

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'Made by Team DATAMRK'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 1px;
            }
            # header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
