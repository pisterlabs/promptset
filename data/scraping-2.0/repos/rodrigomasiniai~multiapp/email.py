import streamlit as st
import openai

st.set_page_config(page_title="AI writer assistant", page_icon="img/Oxta_MLOpsFactor_logo.png", layout="wide")
st.markdown('''<style>.css-1egvi7u {margin-top: -4rem;}</style>''',
    unsafe_allow_html=True)
hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
# Design hide "made with streamlit" footer menu area
hide_streamlit_footer = """<style>#MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)
st.header('Write my email')
with st.sidebar:
    st.markdown(
        "## How to use\n"
        "1. Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  # noqa: E501
        "2. Upload a pdf, docx, or txt file📄\n"
        "3. Get your the document summarization💬\n"
    )
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Paste your OpenAI API key here (sk-...)",
        help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
        value=st.session_state.get("OPENAI_API_KEY", ""),
    )
    openai.api_key = api_key

def gen_mail_contents(email_contents):

    # iterate through all seperate topics
    for topic in range(len(email_contents)):
        input_text = email_contents[topic]
        rephrased_content = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Rewrite the text to be elaborate and polite.\nAbbreviations need to be replaced.\nText: {input_text}\nRewritten text:",
            # prompt=f"Rewrite the text to sound professional, elaborate and polite.\nText: {input_text}\nRewritten text:",
            temperature=0.8,
            max_tokens=len(input_text)*3,
            top_p=0.8,
            best_of=2,
            frequency_penalty=0.0,
            presence_penalty=0.0)

        # replace existing topic text with updated
        email_contents[topic] = rephrased_content.get("choices")[0]['text']
    return email_contents


def gen_mail_format(sender, recipient, style, email_contents):
    # update the contents data with more formal statements
    email_contents = gen_mail_contents(email_contents)
    # st.write(email_contents)  # view augmented contents

    contents_str, contents_length = "", 0
    for topic in range(len(email_contents)):  # aggregate all contents into one
        contents_str = contents_str + f"\nContent{topic+1}: " + email_contents[topic]
        contents_length += len(email_contents[topic])  # calc total chars

    email_final_text = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Write a professional email sounds {style} and includes Content1 and Content2 in that order.\n\nSender: {sender}\nRecipient: {recipient} {contents_str}\n\nEmail Text:",
        # prompt=f"Write a professional sounding email text that includes all of the following contents separately.\nThe text needs to be written to adhere to the specified writing styles and abbreviations need to be replaced.\n\nSender: {sender}\nRecipient: {recipient} {contents_str}\nWriting Styles: motivated, formal\n\nEmail Text:",
        temperature=0.8,
        max_tokens=contents_length*2,
        top_p=0.8,
        best_of=2,
        frequency_penalty=0.0,
        presence_penalty=0.0)

    return email_final_text.get("choices")[0]['text']


def main_gpt3emailgen():

    st.markdown('Generate professional sounding emails based on your comments - powered by GPT')
    st.write('\n')  # add spacing

    st.subheader('\nWhat is your email all about?\n')
    with st.expander("SECTION - Email Input", expanded=True):

        input_c1 = st.text_input('Enter email contents down below! (currently 2x seperate topics supported)', 'topic 1')
        input_c2 = st.text_input('', 'topic 2 (optional)')

        email_text = ""  # initialize columns variables
        col1, col2, col3, space, col4 = st.columns([5, 5, 5, 0.5, 5])
        with col1:
            input_sender = st.text_input('Sender Name', '[me]')
        with col2:
            input_recipient = st.text_input('Recipient Name', '[recipient]')
        with col3:
            input_style = st.selectbox('Writing Style',
                                       ('formal', 'motivated', 'concerned', 'disappointed'),
                                       index=0)
        with col4:
            st.write("\n")  # add spacing
            st.write("\n")  # add spacing
            if st.button('Generate Email'):
                with st.spinner():

                    input_contents = []  # let the user input all the data
                    if (input_c1 != "") and (input_c1 != 'topic 1'):
                        input_contents.append(str(input_c1))
                    if (input_c2 != "") and (input_c2 != 'topic 2 (optional)'):
                        input_contents.append(str(input_c2))

                    if (len(input_contents) == 0):  # remind user to provide data
                        st.write('Please fill in some contents for your message!')
                    if (len(input_sender) == 0) or (len(input_recipient) == 0):
                        st.write('Sender and Recipient names can not be empty!')

                    if (len(input_contents) >= 1):  # initiate gpt3 mail gen process
                        if (len(input_sender) != 0) and (len(input_recipient) != 0):
                            email_text = gen_mail_format(input_sender,
                                                         input_recipient,
                                                         input_style,
                                                         input_contents)
    if email_text != "":
        st.write('\n')  # add spacing
        st.subheader('\nYou sound incredibly professional!\n')
        with st.expander("SECTION - Email Output", expanded=True):
            st.markdown(email_text)  #output the results

main_gpt3emailgen()