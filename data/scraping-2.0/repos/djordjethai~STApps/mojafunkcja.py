import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit as st
import streamlit_authenticator as stauth
from langchain.callbacks.base import BaseCallbackHandler
from io import StringIO
import re


class StreamlitRedirect:
    def __init__(self):
        self.output_buffer = StringIO()

    def write(self, text):
        cleaned_text = re.sub(r'\x1b[^m]*m|[^a-zA-Z\s]', '', text)
        self.output_buffer.write(cleaned_text + "\n")  # Store the output

    def get_output(self):
        return self.output_buffer.getvalue()


# ver 17.08.23


def tiktoken_len(text):
    import tiktoken
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def pinecone_stats(index):
    import pandas as pd
    index_stats_response = index.describe_index_stats()
    index_stats_dict = index_stats_response.to_dict()
    st.subheader("Status indexa:")
    st.write("embedings1")
    flat_index_stats_dict = flatten_dict(index_stats_dict)

    # Extract header and content from the index
    header = [key.split('_')[0] for key in flat_index_stats_dict.keys()]
    content = [key.split('_')[1] if len(key.split('_')) >
               1 else '' for key in flat_index_stats_dict.keys()]

    # Create a DataFrame from the extracted data
    df = pd.DataFrame({'Header': header, 'Content': content,
                      'Value': list(flat_index_stats_dict.values())})

    # Set the desired number of decimals for float values
    pd.options.display.float_format = '{:.2f}'.format

    # Apply formatting to specific columns using DataFrame.style
    styled_df = df.style.apply(lambda x: ['font-weight: bold' if i == 0 else '' for i in range(len(x))], axis=1) \
        .format({'Value': '{:.0f}'})

    # Display the styled DataFrame as a table using Streamlit
    st.write(styled_df)


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def print_nested_dict_st(d):
    for key, value in d.items():
        if isinstance(value, dict):
            st.write(f"{key}:")
            print_nested_dict_st(value)
        else:
            st.write(f"{key}: {value}")


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        token = token.replace('"', '').replace(
            '{', '').replace('}', '').replace('_', ' ')
        self.text += token
        self.container.success(self.text)

    def reset_text(self):
        self.text = ""

    def clear_text(self):
        self.container.empty()


def open_file(filepath):
    with open(filepath, "r", encoding="utf-8") as infile:
        sadrzaj = infile.read()
        infile.close()
        return sadrzaj


def st_style():
    hide_streamlit_style = """
                <style>
                MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                #root > div:nth-child(1) > div > div > div > div > \
                section > div {padding-top: 1rem;}

                [data-testid="stSidebar"] { 
                    background: linear-gradient(to right, #f5f5f5, #f0f0f0); 
                    box-shadow: 15px 15px 10px rgba(0,0,0,0.1);
                    border-radius: 10px;
                }   
                [data-testid="stSidebarNav"] {
                    background-color: #ffffff;
                    font-weight: bold;
                    margin-top: -6rem;
                    border-style: solid;
                    border-width: 20px;
                    border-image: linear-gradient(to right, #f5f5f5, #f0f0f0) 1;
                }

                .stSlider {
                    background-color: #faf5ff;
                    border: 3px solid #c893fc; 
                    border-radius: 10px;
                    padding: 25px;
                    box-shadow: 5px 5px 10px rgba(0,0,0,0.4);
                }
                .stSlider:hover {
                    background-color: #ede2ff;
                    border: 3px solid #9e70f5; 
                }

                [data-testid="stWidgetLabel"] p {
                    font-weight: bold;
                    font-size: 16px;
                }

                [data-testid="stDownloadButton"] button {
                    border: 3px solid #B3A2E9; 
                    background-color: #ECE6FC;
                    box-shadow: 3px 3px 10px rgba(0,0,0,0.4);
                }
                [data-testid="stDownloadButton"] button:hover {
                    border: 3px solid #8F78CD;
                    background-color: #CFC8F2;
                }

                [data-testid="stFormSubmitButton"] button {
                    border: 3px solid #B3A2E9; 
                    background-color: #ECE6FC;
                    box-shadow: 3px 3px 10px rgba(0,0,0,0.4);
                }
                [data-testid="stFormSubmitButton"] button:hover {
                    border: 3px solid #8F78CD;
                    background-color: #CFC8F2;
                }

                [data-testid="stFileUploader"] {
                    border: 3px solid #72B6FC; 
                    background-color: #eff6ff;
                    padding: 10px;
                    border-radius: 10px;
                }
                [data-testid="stFileUploader"]:hover {
                    border: 3px solid #4D8FD6;
                    background-color: #DCE4F7;
                }

                [data-testid="stCaptionContainer"] {
                    textColor: white;
                }
                
                [data-testid="baseButton-header"] {
                    background-color: #DCE4F7;
                }
                
                [data-testid="stFileUploader"] {
                    border: 3px solid #72B6FC; 
                    background-color: #eff6ff;
                    padding: 10px;
                    border-radius: 10px;
                    box-shadow: 5px 5px 4px rgba(0,0,0,0.2);
                }
                [data-testid="stFileUploader"]:hover {
                    border: 3px solid #4D8FD6;
                    background-color: #DCE4F7;
                }

                [data-testid="stFileUploader"] button {
                    border: 3px solid #B3A2E9; 
                    background-color: #ECE6FC;
                    box-shadow: 3px 3px 10px rgba(0,0,0,0.4);
                }
                [data-testid="stFileUploader"] button:hover {
                    border: 3px solid #8F78CD;
                    background-color: #CFC8F2;
                }

                [data-testid="stForm"] {
                    border: 3px solid #4ac9a2; 
                    background-color: #ecfdf5; 
                    box-shadow: 5px 5px 4px rgba(0,0,0,0.2);
                }
                [data-testid="stForm"]:hover {
                    border: 3px solid #51B391;
                    background-color: #c5f2e1;
                }

                [data-baseweb="select"] {
                    font-weight: bold;
                    font-size: 16px;
                    box-shadow: 5px 5px 4px rgba(0,0,0,0.5);
                }
                [data-testid="stWidgetLabel"] [data-baseweb="textarea"] {
                    border: 3px solid #51B391;
                    box-shadow: 5px 5px 4px rgba(0,0,0,0.2);
                }
                [data-testid="stExpander"] {
                    border: 2px solid #7EB6F6; 
                    border-radius: 10px;
                    box-shadow: 5px 5px 10px rgba(0,0,0,0.4);
                }
                
                [data-baseweb="select"] [data-baseweb="icon"] {
                    border: 3px solid #eff6ff;
                    background-color: #eff6ff; 
                    box-shadow: 2px 2px 4px rgba(0,0,0,0.4);
                }

                [data-baseweb="select"] [data-baseweb="icon"]:hover {
                    border: 3px solid #d8e3f3;
                    background-color: #d8e3f3; 
                    box-shadow: 2px 2px 4px rgba(0,0,0,0.4);
                }

                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def positive_login(main, verzija):

    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
            config['preauthorized'],
        )

        name, authentication_status, username = authenticator.login(
            'Login to Positive Apps', 'main')

    # Get the email based on the name variable
        email = config['credentials']['usernames'][username]['email']
        access_level = config['credentials']['usernames'][username]['access_level']
        st.session_state["name"] = name
        st.session_state["email"] = email
        st.session_state["access_level"] = access_level

    if st.session_state["authentication_status"]:
        with st.sidebar:
            st.caption(f"{verzija}")
            authenticator.logout('Logout', 'main', key='unique_key')
        # if login success run the program
        main()
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')

    return name, authentication_status, email

# define model and temperature


def init_cond_llm():
    with st.sidebar:
        x1 = st.text("", help="Modeli se razlikuju po kvalitetu, brzini i ceni upotrebe.")
        model = st.selectbox(
            "Odaberite model", label_visibility="collapsed", options=("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"))
        x2 = st.text("")
        x3 = st.text("", help="Temperatura utice na kreativnost modela. Sto je veca temperatura, model je kreativniji, ali i manje pouzdan.")
        temp = st.slider(
            'Set temperature (0=strict, 1=creative)', 0.0, 2.0, step=0.1)
    return model, temp
# error handling on Serbian


def greska(e):
    if "maximum context length" in str(e):
        st.warning(
            f"Nisam u mogucnosti za zavrsim tekst. Pokusajte sa modelom koji ima veci kontekst.")
    elif "Rate limit" in str(e):
        st.warning(
            f"Nisam u mogucnosti za zavrsim tekst. Broj zahteva modelu prevazilazi limite, pokusajte ponovo za nekoliko minuta.")
    else:
        st.warning(
            f"Nisam u mogucnosti za zavrsim tekst. Pokusajte ponovo za nekoliko minuta. Opis greske je {e}")
