from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.chat_models import ChatOpenAI

import pandas as pd
import streamlit as st
import random
from datetime import datetime
import time
from user_authentication import UserAuthentication, is_key_valid
from streamlit_database import DBConnection
from constants import prompt_strategies

st.set_page_config(layout="wide")

# CSS for some layout settings
centered_large_bold_css = """
<style>
.centered-text-large-bold {
    text-align: center;
    font-weight: bold;
    font-size: 20px; /* You can adjust the size as needed */
    margin-top: 3px; /* Add some top margin */
    margin-bottom: 5px; /* Add some bottom margin */
}
</style>
"""

bottom_button_css = """
<style>
div.stButton {
    margin-top: 13px;
}
</style>
"""

scrollable_textarea_css = """
<style>
/* Targeting the text area widget */
.stTextArea > div > div {
    overflow-y: scroll; /* Vertical scroll bar */
}
</style>
"""

# initialize the session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None

# initialize the session state for page after login
if 'option_function' not in st.session_state:
    st.session_state.option_function = 'Get Help from GPT'

# initialize the session state for function 2
if 'random_select' not in st.session_state:   # for random select, 0 for no, 1 for yes
    st.session_state.random_select = 0

# initialize the session state for function 1
if 'captions' not in st.session_state:
    st.session_state.captions = []
if 'descriptions' not in st.session_state:
    st.session_state.descriptions = []
if 'methods' not in st.session_state:
    st.session_state.methods = []
if 'num_cleared' not in st.session_state:   # for clear history, the number of cleared records
    st.session_state.num_cleared = 0
if 'draft_val' not in st.session_state:   # for scratch paper, the value of the scratch paper
    st.session_state.draft_val = ''
if 'descp' not in st.session_state:   # for description, the value of the description
    st.session_state.descp = ''
if 'descp_type' not in st.session_state:   # for description type, 0 for input, 1 for GPT4, 2 for llava
    st.session_state.descp_type = 0

# extract the valid contest number from the database as a list
contest_num_list = DBConnection.contest_num_list()
lastest_contest_num = max(contest_num_list)   # get the lastest contest number

# default descriptions for the image in the sharing event.
two_descriptions = DBConnection.select(f"""SELECT llava_descp, gpt_descp FROM new_descriptions WHERE contest_num={lastest_contest_num}""")
default_descps = [
    two_descriptions.iloc[0, 0],
    two_descriptions.iloc[0, 1]
]

##################################### main #####################################
# if the user is not logged in, show the login page; otherwise, show the main page
if st.session_state['logged_in'] == False:
    UserAuthentication.user_account()
else:
    # function 2: get inspiration
    if st.session_state.option_function == 'Inspiration':
        # set up the page layout
        col1, col2 = st.columns([1,2], gap="medium")
        with col1:
            st.selectbox('Choose a function', ('Get Help from GPT', 'Inspiration'), key="option_function")
            col1_1, col1_2 = st.columns([3.5,1], gap="medium")
            with col1_2:
                log_out = st.button("Log out")

            if log_out:
                UserAuthentication.logout_user()
                st.info("You have been logged out.")
                time.sleep(1)
                st.rerun()

        with col2:
            st.title('Get some inspiration from previous cartoons! :bulb:')
            # create a select box for contest number from 510 to 863 but not 525, 643, 646, 655 and a random option
            options = contest_num_list
            options.insert(0, 'Random')
            contest_num = st.selectbox('Choose a random contest number or a specific one', options)

            if st.button(':red[Give me some inspiration]'):
                st.session_state.random_select = 0
                
                # if contest number is random, give a random number from 510 to 863 but not 525, 643, 646, 655
                if contest_num == 'Random':
                    contest_num = random.choice(options[1:])
                    st.session_state.random_select = 1
                
                # get the image and caption from the database
                random_integers = [random.randint(0, 19) for _ in range(3)]
                inspiration_URL = DBConnection.select(f"""SELECT image_url FROM base Where contest_num={contest_num}""")
                inspiration_caption = DBConnection.select(f"""SELECT caption FROM result WHERE contest_num={contest_num}""")
                col2_1, col2_2 = st.columns([0.4, 0.6], gap="large")
                with col2_1:
                    st.image(inspiration_URL.iloc[0, 0], width=360)
                with col2_2:
                    st.write("**Funny Caption 1:**")
                    st.write(inspiration_caption.iloc[random_integers[0], 0])
                    st.write("**Funny Caption 2:**")
                    st.write(inspiration_caption.iloc[random_integers[1], 0])
                    st.write("**Funny Caption 3:**")
                    st.write(inspiration_caption.iloc[random_integers[2], 0])

                # insert the record into the database
                DBConnection.insert(f"""INSERT INTO interface_records (random_select, contest_num, used_function, time, user_id) VALUES (%s, %s, %s, %s, %s)""",
                                    (st.session_state.random_select, contest_num, 'Inspiration', datetime.now(), st.session_state.user_id))

     

    # function 1: get help from GPT
    if st.session_state.option_function == 'Get Help from GPT':
        
        # set up the page layout
        col1, col2 = st.columns([1,2], gap="medium")
        with col1:
            st.selectbox('Choose a function', ('Get Help from GPT', 'Inspiration'), key="option_function")

            with st.form("my_form"):
                st.text_input('Your GPT API:', key="API", type="password")
                model_selection = st.selectbox('Choose an AI', ('gpt-4',  'gpt-3.5-turbo'), key="model")
                col1_1, col1_2 = st.columns([4,1], gap="medium")
                with col1_1:
                    Key_submit = st.form_submit_button("Submit")
                with col1_2:
                    log_out = st.form_submit_button("Log out")

            if Key_submit:
                # check if the API key is valid
                if is_key_valid(st.session_state.API):
                    st.success("API key is valid!")
                else:
                    st.error("API key is invalid!")
                    st.stop()
                
            if log_out:
                UserAuthentication.logout_user()
                st.info("You have been logged out.")
                time.sleep(1)
                st.rerun()   # rerun the app
                
        
        # after submitting the API key, show the main page
        if not st.session_state.API:
            with col2:
                st.title(':red[Please enter the API]')
        else:
            # create a chat box
            msgs = StreamlitChatMessageHistory(key="langchain_messages")
            memory = ConversationBufferMemory(chat_memory=msgs)
            template = """You are an AI chatbot having a conversation with a human.

            {history}
            Human: {human_input}
            AI: """
            prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
            llm_chain = LLMChain(llm=ChatOpenAI(openai_api_key=st.session_state.API, model=st.session_state.model), prompt=prompt, memory=memory)

            # create prompt templates
            prompt_initial = PromptTemplate(
                input_variables=["description", "caption", "suggestion_method"],
                template="You are assisting someone trying to think of funny captions for a New Yorker cartoon. \
                    Here is a description of the cartoon image: {description}. \
                    So far the user has come up with the following captions: {caption}. \
                    Your job is to help the user come up with better captions. \
                    One way of making a caption funnier is to {suggestion_method}. \
                    With this in mind, please talk directly to the user to suggest ways the user can adapt their ideas to create a funnier caption. \
                    Respond in no more than three sentences, and avoid repeating any previous advice.")
            prompt_for_descp_method_change = PromptTemplate(input_variables=["description", "suggestion_method"], 
                                                            template="Based on the new description: {description}. \
                                                                    And another way of making a caption funnier is to {suggestion_method}. \
                                                                    Please give the user some more advice.")
            prompt_for_cap_method_change = PromptTemplate(input_variables=["caption", "suggestion_method"], 
                                                            template="From your prior advice, the user added the following captions: {caption}. \
                                                                And another way of making a caption funnier is to {suggestion_method}. \
                                                                Please give the user some more advice.")
            prompt_for_method_change = PromptTemplate(input_variables=["suggestion_method"], 
                                                        template="Based on the new method for improve funniness: {suggestion_method}, \
                                                            Please give the user some more suggestions.")
            prompt_complete = PromptTemplate(
                input_variables=["description", "caption", "suggestion_method"],
                template="From your prior advice, the user updated the description: {description}. \
                    the user added the following captions: {caption}.\
                    Now the user would like some more advice. Another way of making a caption funnier is to {suggestion_method}. \
                    With this in mind, please talk directly to the user to suggest new ways the user can adapt their ideas to create a funnier caption. \
                    Respond in no more than three sentences, and avoid repeating any previous advice.")
            
            # history showing parts
            with col1:
                with st.expander("View your records:"):
                    # clear history but not the chat
                    if st.button("Clear history"):
                        st.session_state.num_cleared = len(st.session_state.captions)
                        st.rerun()   # rerun the app
                    for i in reversed(range(len(st.session_state.captions) - st.session_state.num_cleared)):
                        rid = i+1
                        st.write(f"**Record {rid}:** ")
                        st.write(f"*Description:* \n\n\n {st.session_state.descriptions[i+st.session_state.num_cleared]}")
                        # change the list of captions to a string and for each caption, add a new line
                        captions_str = '\n\n\n'.join(st.session_state.captions[i+st.session_state.num_cleared])
                        st.write("*New added captions:*" + '\n\n\n' + captions_str)
                        st.write("*AI suggestion:*" + '\n\n\n' + msgs.messages[(i+st.session_state.num_cleared)*2+1].content)
    
            with col2:
                col2_1, col2_2 = st.columns([0.6,1], gap="medium")
                # create the showed cartoon and a scratch paper
                with col2_1:
                    st.markdown(centered_large_bold_css, unsafe_allow_html=True)
                    st.markdown("<div class='centered-text-large-bold'>The Newest New York Cartoon</div>", unsafe_allow_html=True)   # create a centered title for the cartoon
                    st.image(f'https://nextml.github.io/caption-contest-data/cartoons/{lastest_contest_num}.jpg')
                with col2_2:
                    st.markdown(scrollable_textarea_css, unsafe_allow_html=True)
                    current_draft = st.text_area('Free Scratch Paper', height=370, value=st.session_state.draft_val, 
                                                help="This draft paper is for recording captions you come up. \
                                                    You can use 'Record to draft' button to record your caption to the paper.")

                # create a form for getting help from GPT
                with st.form(key='my_form3'):
                    # create a text input for the description
                    col2_9, col2_10 = st.columns([6,1])   # layout
                    with col2_9:
                        descp = st.text_area('Please describe the content of the cartoon in as much detail as possible:', 
                                            value=st.session_state.descp, height=150, help="You can use 'Autofill description' button to autofill the description or enter your own description.")
                    with col2_10:
                        st.markdown(bottom_button_css, unsafe_allow_html=True)
                        autofill_descp = st.form_submit_button(label="Autofill description")   # button for autofilling the description
                    # create caption text input and record to scratch button
                    col2_5, col2_6 = st.columns([6,1])   # layout
                    with col2_5:
                        cap = st.text_input('Write your caption:')
                    with col2_6:
                        st.markdown(bottom_button_css, unsafe_allow_html=True)
                        reocrd_to_draft_button = st.form_submit_button(label='Record to draft', help='Click to put your current caption to the draft paper')   # button for recording the caption to the scratch paper
                    # create help button and reset button
                    col2_7, col2_8 = st.columns([1,4])   # layout
                    with col2_7:
                        help_button = st.form_submit_button(label='Ask for assistance')   # button for asking for assistance
                    with col2_8:
                        reset_button = st.form_submit_button(label='Reset chat')   # button for resetting the chat
                

                # run for autofilling the description
                if autofill_descp:
                    default_descp = random.choice(default_descps)
                    # reocred the description type used
                    st.session_state.descp = default_descp
                    if default_descp == default_descps[0]:
                        st.session_state.descp_type = 1
                    elif default_descp == default_descps[1]:
                        st.session_state.descp_type = 2
                    st.rerun()

                # run for recording the caption to the scratch paper
                if reocrd_to_draft_button:
                    if cap == '':
                        st.error('Please enter the caption!')
                        st.stop()
                    else:
                        st.session_state.draft_val = st.session_state.draft_val + cap + '\n'
                        st.rerun()
                
                # keep last message showing even after rerun
                if help_button == False and len(msgs.messages) > 0:
                    st.chat_message("ai").write(msgs.messages[-1].content)
                
                # run for asking for assistance
                if help_button:
                    # randomly choose a suggestion method
                    if len(st.session_state.methods) == 0:   # if it is the first time to ask for help
                        option_help = random.choice(prompt_strategies)
                    else:
                        # avoid repeating the suggestion method in continuous two times
                        option_help = random.choice(prompt_strategies)
                        while option_help == st.session_state.methods[-1]:
                            option_help = random.choice(prompt_strategies)

                    # check if any input is empty
                    if descp == '':
                        st.error('Please enter the description!')
                        st.stop()
                    else:
                        st.session_state.descp = descp

                    if cap == '' and st.session_state.draft_val == '':
                        st.error('Please enter the caption!')
                        st.stop()

                    # process newly added captions in the draft paper
                    if st.session_state.draft_val == '':
                        st.session_state.draft_val = st.session_state.draft_val + cap + '\n'   # add the caption to the draft paper
                        list_draft_caps = [cap]
                    else:
                        list_draft_caps = st.session_state.draft_val.split('\n')   # change the all value of the draft paper to list
                        # remove the empty items in the list
                        while '' in list_draft_caps:
                            list_draft_caps.remove('')
                        if cap != list_draft_caps[-1]:
                            list_draft_caps.append(cap)
                            st.session_state.draft_val = st.session_state.draft_val + cap + '\n'   # add the caption to the draft paper
                    # for first time to ask for help, all captions in the draft paper are new added captions
                    if len(st.session_state.captions) == 0:
                        new_add_caps = list_draft_caps
                    else:
                        # merge all lists of st.session_state.captions into one list
                        all_previous_caps = []
                        for i in range(len(st.session_state.captions)):
                            all_previous_caps = all_previous_caps + st.session_state.captions[i]
                        new_add_caps = list(set(list_draft_caps) - set(all_previous_caps))   # new_add_caps is all captions different from the all previous record
                        new_add_caps = [cap.strip() for cap in new_add_caps]   # remove spaces at the beginning and end of each caption

                    # add the caption, description to the captions, descriptions history
                    st.session_state.descriptions.append(descp)
                    st.session_state.captions.append(new_add_caps)
                    st.session_state.methods.append(option_help)

                    # run the prompts
                    if len(msgs.messages) == 0:   # running the prompt for the first time
                        prompt = prompt_initial.format(description=descp, caption=new_add_caps, suggestion_method=option_help)  
                        response = llm_chain.run(prompt)
                        st.chat_message("ai").write(response)
                        change_type = 0
                    else:
                        # if the description and suggestion method are changed
                        if st.session_state.descriptions[-2] != descp and len(new_add_caps) == 0:  
                            prompt = prompt_for_descp_method_change.format(description=descp, suggestion_method=option_help)
                            response = llm_chain.run(prompt)
                            st.chat_message("ai").write(response)
                            change_type = 1
                        # if the caption and suggestion method are changed
                        elif len(new_add_caps) > 0 and st.session_state.descriptions[-2] == descp:  
                            prompt = prompt_for_cap_method_change.format(caption=new_add_caps, suggestion_method=option_help)
                            response = llm_chain.run(prompt)
                            st.chat_message("ai").write(response)
                            change_type = 2
                        # if only the suggestion method is changed    
                        elif st.session_state.methods[-2] != option_help and st.session_state.descriptions[-2] == descp and len(new_add_caps) == 0:  
                            prompt = prompt_for_method_change.format(suggestion_method=option_help)
                            response = llm_chain.run(prompt)
                            st.chat_message("ai").write(response)
                            change_type = 3
                        # if two or more items are changed
                        else:
                            prompt = prompt_complete.format(description=descp, caption=new_add_caps, suggestion_method=option_help)  
                            response = llm_chain.run(prompt)
                            st.chat_message("ai").write(response)
                            change_type = 4
                    
                    new_add_caps_str = '|||'.join(new_add_caps)  # change the list of new_add_caps to a string
                    # insert the record into the database
                    DBConnection.insert(query=f"""INSERT INTO interface_records (description, caption, method, used_function, change_type, user_id, contest_num, model, time, suggestion, description_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""", 
                                        data=(descp, new_add_caps_str, option_help, 'Get Help from GPT', change_type, st.session_state.user_id, lastest_contest_num, st.session_state.model, datetime.now(), response, st.session_state.descp_type))
                    
                    st.session_state.descp_type = 0   # reset the description type
                    st.rerun()   # rerun the app for showing the updated draft paper
                    
                # reset the chat
                if reset_button:
                    # clean history
                    st.session_state.descriptions = []
                    st.session_state.captions = []
                    st.session_state.methods = []
                    st.session_state.draft_val = ''
                    st.session_state.descp = ''
                    st.session_state.descp_type = 0
                    # clean chat
                    del st.session_state.langchain_messages
                    msgs = StreamlitChatMessageHistory(key="langchain_messages")
                    st.rerun()   # rerun the app

