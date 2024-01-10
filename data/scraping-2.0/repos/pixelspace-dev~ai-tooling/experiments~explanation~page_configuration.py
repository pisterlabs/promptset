import streamlit as st
from langchain.memory import ConversationBufferMemory

from summarization import reset_chat, display_percentage, prompt_change, summarize
from token_counter import calculate_tokens_used

def define_variables():
    st.set_page_config(layout="wide",)

    # add variables to the session state so AI can remember what has been said
    if 'user_message' not in st.session_state:
        st.session_state.user_message = []
    if 'ai_message' not in st.session_state:
        st.session_state.ai_message = []
    if 'set_new_prompt' not in st.session_state:
        st.session_state.set_new_prompt = False
    if 'prompt' not in st.session_state:
        st.session_state.prompt = ""
    if 'response' not in st.session_state:
        st.session_state.response = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
    if 'correct_password' not in st.session_state:
        st.session_state.correct_password = False
    ### temporary
    # if 'partial_summaries' not in st.session_state:
    #     st.session_state.partial_summaries = []
    ###



def enter_password():
    st.write("Enter Password:")
    with st.form("password"):
        password_col1, password_col2 = st.columns([2,1])
        with password_col1:
            entered_password = st.text_input(label="Enter Password:", label_visibility="collapsed", placeholder="password123", type="password")
        with password_col2:
            if st.form_submit_button(label=":green[Check]") and entered_password:
                if entered_password == st.secrets["PASSWORD"]:
                    st.session_state.correct_password = True
                else:
                    st.session_state.correct_password = False
                    with password_col1:
                        st.error("incorrect password")



def file_input_configuration(explain_placeholder, model, guide, document_size, summary_size, document_type):
    if document_type == "PDF":
        beginning_page = st.number_input("First Page:", step=1, value=1)
        last_page = st.number_input("Last Page:", step=1, value=2)
    else:
        beginning_page = 1
        last_page = 2
    st.file_uploader(label="file", label_visibility="collapsed", key="file")

    input_col1, input_col2, input_col3= st.columns([3,1.2,3])
    if input_col2.form_submit_button(":green[Submit]"):
        with explain_placeholder:
            summarize(model, guide, beginning_page, last_page, document_size, summary_size, document_type)
            ### temporary
        # with intermediate_summary_placeholder:
        #     for sum in st.session_state.partial_summaries:
        #         st.markdown(sum)
        #     del st.session_state.partial_summaries



def guide_configuration():
    guide = st.text_area(label="Summary guidelines", label_visibility="collapsed")
    guide_col1, guide_col2, guide_col3 = st.columns([3,1.4,3])
    guide_col2.form_submit_button(":green[Set]", on_click= prompt_change(guide))
    return guide



def sidebar_configuration():
    with st.sidebar:
        st.subheader("Select Model")
        model = st.selectbox(label="Select Model", label_visibility="collapsed", options=["gpt-4", "gpt-3.5-turbo-16k"])

        st.subheader("Document Size", help="Factors such as font size can effect the maximum allowed page count for small documents")
        document_size = st.selectbox(label="Select Document Size", 
                                    label_visibility="collapsed", 
                                    options=["small ( < 10 pages or 8,000 tokens )", "large ( > 10 pages or 8,000 tokens )"],
                                    )

        summary_size = st.slider(label="Select Summary Detail", 
                                min_value=100,
                                max_value= 3000, 
                                value=3000, 
                                step=10,
                                help="""A higher value allows for more detail, slider only applies to long documents (experimental)""")

    return model, document_size, summary_size



def tokens_used_configuration(model):
    (tokens_used, percentage) = calculate_tokens_used(model)

    st.subheader(f"Tokens Used: {tokens_used}", help= "This does not include tokens from intermediate summaries with large documents")
    st.subheader("Percentage of Tokens Remaining:")

    display_percentage(percentage)

    st.button(label="Clear Chat", on_click=reset_chat)