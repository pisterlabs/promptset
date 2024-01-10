import gpt4_curl, gpt4_tls, gpt3

from time import sleep
from asyncio import run
from langchain.prompts import PromptTemplate
from utils import *
import streamlit as st
from pathlib import Path
from streamlit_option_menu import option_menu

question_prompt_template = """
    You are very good at handling very long texts,so I will give you a video transcription (mostly,in english) splitted in small pieces.This is piece number {i}
        You will get a request about it, you should translate this request to the language of transcription if it isn't,
        then give me the answer in the language of question, or in another one if I asked.
        you should enhance your answer with useful proper resources.\n\n
        
        transcription: {input} \n\n
        
        request: {question} \n\n
        
        feel free to neglect the given transcription if you see that the request is not related to it like thank you, hello or ok and similars, provide instead an appropriate answer like you are welcome.
        
        you may be asked to provide your answer in specific language like arabic, and you must provide your answer in the asked language.
            
        Your answer:\n\n
    """
old = '''
You are very good at handling very long texts,so I will give you a video transcription splitted in small pieces,this is piece number {i}.You will get a query about it,\n\n
    transcription: {input}\n\n
    
    query: {question}    \n\n
    feel free to neglect the given transcription if you see that the query is not related to it like thank you or ok and similars, provide instead an appropriate answer like you are welcome.
    query may be a question about it or not, do your best to extract the answer if it exists or make up a suitable answer but hint me if you made one(say for example This answer is not mentioned but and this is a made one).
    or it can be explaining something in a simpler way,
    or it can be writing programming code explaining a concept in it,
    or summerizing it in number of words,
    or splitting it to chapters of homogenious content like youtube does.Do your best to give me the answer in this format "hr:min:sec title" and make sure that each chapter is at least 3 minutes.
    or any query
    you may be asked to provide your answer in specific language like arabic, and you must provide your answer in the asked language.
    Also you may be provided with the previous query and a summary of your answer to use them like a memory of past interactions.
    You can neglect them if you see that the answer of the current query doesn't need them.
        
    Your answer:\n\n
'''
prompt = PromptTemplate(input_variables=["i","input", "question"], template=question_prompt_template)

async def get_answer_from_chatgpt(question):
    try:
        resp = await gpt3.Completion().create(question)
        return resp
    except:
        st.info('Service may be stopped or you are disconnected with internet. Feel free to open an issue here "https://github.com/Mohamed01555/VideoQuERI"')
        st.stop()
# Fetching answers from You.com will only work offline
async def get_answer_from_youbot(question):
    try:
        resp = await gpt4_curl.Completion().create(question)
        return resp
    except:
        try:
            resp = await gpt4_tls.Completion().create(question)
            return resp
        except:
            st.info('Service may be stopped or you are disconnected with internet. Feel free to open an issue here "https://github.com/Mohamed01555/VideoQuERI"')
            st.stop()

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def main():
    # setup streamlit page
    st.set_page_config(
        page_title="VideoQuERI",
        page_icon="vqueri.jpeg")
    
    option = option_menu(
    menu_title=None,
    options=["Home", "FAQs", "Contact", "Donate"],
    icons=["house-check", "patch-question-fill", "envelope","currency-dollar"],
    orientation='horizontal',
    styles={
        "container": {"padding": "0!important", "background-color": "#333"},        
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#ff9900"},
        "nav-link-selected": {"background-color": "#6c757d"},
    }
    )   

    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.markdown(html_code, unsafe_allow_html=True)

    # initialize responses.
    if "responses" not in st.session_state:
        st.session_state.responses = []
    
    # initialize caption.
    if "caption" not in st.session_state:
        st.session_state.caption = None

    # initialize test_splitter.
    if "text_splitter" not in st.session_state:
        text_splitter = None
    
    # Initialize session state variables
    if 'captions' not in st.session_state:
        st.session_state.captions = {}

    # initialize chunks.
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    
    if "button_pressed" not in st.session_state:
        st.session_state.button_pressed = False
    
    if "chosen_chunks" not in st.session_state:
        st.session_state.chosen_chunks = []

    if "prev_qa" not in st.session_state:
        st.session_state.prev_qa = None
    
    if 'video_url_list' not in st.session_state:
        st.session_state.video_url_list = []

    if "question" not in st.session_state:
        st.session_state.question = None

    if "chosen_radio" not in st.session_state:
        st.session_state.chosen_radio = None

    if "chosen_model" not in st.session_state:
        st.session_state.chosen_model = None

    # Set the maximum number of stored captions
    MAX_CAPTIONS = 10

    with st.sidebar:
        video_url = st.text_input("**Paste the video url here:**")
        
        help_slider= "Processing the entire video in a single iteration might be beyond the capability of GPT.\
                So we split it in chunks. Please choose the desired chunk size. The bigger the chunk size is, the more precise the answer you get."
        selected_value = st.slider("Select a value for chunk size", min_value=100, max_value=3000, value=1500, step=1, help=help_slider)
        
        help_button = "Creating captions from scratch for a video lasting one hour typically requires approximately 2 minutes.\n \
                       In the event of the server experiencing a high volume of requests, the caption generation process could become significantly delayed.\
                       If this occurs, we recommend revisiting at a different time. Alternatively, if you already possess the caption, please feel free to provide it below."
        
        if st.button("Generate the Caption...", help = help_button):
            st.session_state.button_pressed = True
            if (video_url.strip().startswith('http') or video_url.strip().startswith('https')):
                with st.spinner("Generating the video Caption..."):
                    if video_url not in st.session_state.captions.keys():
                        st.session_state.caption, ret = get_transcript(video_url)
                        
                        if st.session_state.caption:
                            if ret == 'return_from_whisper':
                                st.session_state.captions[video_url] = st.session_state.caption
                            text_splitter = TokenTextSplitter(chunk_size = selected_value, chunk_overlap=11)
                            st.session_state.chunks = text_splitter.split_documents(st.session_state.caption)
                            
                            #add the url to the list to ensure whether i will provide a summary of perious qa                            
                            st.info("Caption was generated successfully. You can ask now.")
                        
                        else:
                            st.info('Most likely it is not a video, Or caption eneration service if full now. Please try again later')
                            st.stop() 
                    else:
                        st.session_state.caption = st.session_state.captions[video_url]
                        text_splitter = TokenTextSplitter(chunk_size = selected_value, chunk_overlap=11)
                        st.session_state.chunks = text_splitter.split_documents(st.session_state.caption)
                    
                        #add the url to the list to ensure whether i will provide a summary of perious qa    
                        st.info("Caption was generated successfully. You can ask now")

                        
                    # Limit the number of stored captions
                    if len(st.session_state.captions) > MAX_CAPTIONS:
                        oldest_url = next(iter(st.session_state.captions))
                        st.session_state.captions.pop(oldest_url)
                
            else:
                st.info('Valid URL must start with `http://` or `https://` ')
                st.stop()
        
        if st.session_state.button_pressed:
            t=''
            for c,doc in enumerate(st.session_state.chunks):        
                start, end = extract_start_end_time(doc.page_content)
                if start is not None and end is not None:  
                    t += f'Chunk {c+1} : from {start} to {end}\n\n'
            with st.expander('**Info :information_source:**'):
                st.info(
                    f'Number of Chunks : {len(st.session_state.chunks)}\n\n{t}'
                    )

            with st.expander("**If your query is about specific chunks, please choose them** :slightly_smiling_face:"):

                st.session_state.chosen_chunks = []
                for i in range(len(st.session_state.chunks)):
                    chosen_chunk = st.checkbox(label= str(i+1))
                    if chosen_chunk:
                        st.session_state.chosen_chunks.append(i + 1)

            if st.session_state.chosen_chunks:
                st.info(f"Selected Chunks: {st.session_state.chosen_chunks}")

        st.session_state.chosen_model = st.radio("Please, choose the backend model", ['YouBot', 'ChatGPT'])
        
        st.session_state.chosen_radio = st.radio("Do you want to add some sort of memory?", ['No', 'Yes'], help="Note that it is not that accurate memory")
           
    if option == 'Home':
        for response in st.session_state.responses:
            with st.chat_message(response['role']):
                st.markdown(response['content'], unsafe_allow_html=True)


        st.session_state.question = st.chat_input('Your Query...')
        if st.session_state.question:
            if not st.session_state.button_pressed:
                st.info("You forgot to enter your Video URL and click *Generate the Caption...* button.")
                st.stop()

            with st.chat_message('user'):
                st.markdown(st.session_state.question,unsafe_allow_html=True)

            st.session_state.responses.append({'role':"user", 'content': st.session_state.question})
            
            with st.chat_message('assistant'):
                st.session_state.message_placeholder = st.empty()
                full_response = ''
                #if the user entered specefic chunks to query about
                if len(st.session_state.chosen_chunks) != 0:
                    for c in st.session_state.chosen_chunks: 
                        doc = st.session_state.chunks[c-1] 
                        # full_response = answer(chunk_number=c, doc = doc, question = question)
                        query = prompt.format(i = c, input = doc.page_content, question = st.session_state.question)
                        
                        try:
                            if video_url == st.session_state.video_url_list[-1]:
                                query += st.session_state.prev_qa if st.session_state.prev_qa else ''
                        except:
                            query = query
                        start, end = extract_start_end_time(doc.page_content)
                        if start is not None and end is not None:  
                            with st.spinner(f"Searching for the answer in the period {start} --> {end}"):
                                if st.session_state.chosen_model == 'YouBot':
                                    ai_response = run(get_answer_from_youbot(query))
                                    ai_response_decoded = decode_unicode(ai_response)
                                else:
                                    ai_response = run(get_answer_from_chatgpt(query))
                                    ai_response_decoded = decode_unicode(ai_response)
                                time_ = f"""<span style="color: #00FF00;">Answer in the period <span style="color: #800080;">{start} --> {end}</span> is \n\n</span>"""
                                full_response += '\n' + time_ + '\n'+ ai_response_decoded + '\n'
  
                                st.session_state.message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                                    
                        
                        else:
                            if st.session_state.chosen_model == 'YouBot':
                                    ai_response = run(get_answer_from_youbot(query))
                                    ai_response_decoded = decode_unicode(ai_response)
                            else:
                                ai_response = run(get_answer_from_chatgpt(query))
                                ai_response_decoded = decode_unicode(ai_response)
                            
                            full_response += '\n\n' + ai_response_decoded + '\n\n'
                        
                            st.session_state.message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                            
                
                #if the user did not entered specefic chunks, use all chunks
                else:
                    for c,doc in enumerate(st.session_state.chunks):     
                        # full_response = answer(chunk_number=c+1, doc = doc, question = question)   
                        query = prompt.format(i = c+1, input = doc.page_content, question = st.session_state.question)
                        
                        try:
                            if video_url == st.session_state.video_url_list[-1]:
                                query += st.session_state.prev_qa if st.session_state.prev_qa else ''
                        except:
                            query = query
                        
                        start, end = extract_start_end_time(doc.page_content)
                        if start is not None and end is not None:  
                            with st.spinner(f"Searching for the answer in the period {start} --> {end}"):
                                if st.session_state.chosen_model == 'YouBot':
                                    ai_response = run(get_answer_from_youbot(query))
                                    ai_response_decoded = decode_unicode(ai_response)
                                else:
                                    ai_response = run(get_answer_from_chatgpt(query))
                                    ai_response_decoded = decode_unicode(ai_response)
                                time = f"""<span style="color: #00FF00;">Answer in the period <span style="color: #800080;">{start} --> {end}</span> is \n\n</span>"""
                                full_response += '\n' + time + '\n'+ ai_response_decoded + '\n'
                                
                                st.session_state.message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                        
                        else:
                            if st.session_state.chosen_model == 'YouBot':
                                    ai_response = run(get_answer_from_youbot(query))
                                    ai_response_decoded = decode_unicode(ai_response)
                            else:
                                ai_response = run(get_answer_from_chatgpt(query))
                                ai_response_decoded = decode_unicode(ai_response)
                            full_response += '\n' + ai_response_decoded
                            
                            st.session_state.message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                
                st.session_state.message_placeholder.markdown(full_response, unsafe_allow_html=True)
                
                if st.session_state.chosen_radio == 'Yes':
                    # get a summary of the answer and append before the next question
                    summary_prompt = f"""
                    Please summarize this in 100 to 200 words as a mximum.
                    Retain any programming code present, even if doing so exceeds the 200-word limit.
                    Capture the entites if exist\n{full_response}
                    """
                    summary = run(get_answer_from_youbot(summary_prompt) if st.session_state.chosen_model == 'YouBot' else get_answer_from_youbot(summary_prompt) )
                    st.session_state.prev_qa = f"This is the previous question: {st.session_state.question}\nand this is the summary of your answer: {summary}"
                

                            
            st.session_state.video_url_list.append(video_url)

            st.session_state.responses.append({'role' : 'assistant', 'content' : full_response})

    elif option == 'FAQs':
        FAQs()
    elif option == 'Contact':
        contact()
    else:
        donate()

if __name__ == '__main__':
    main()
