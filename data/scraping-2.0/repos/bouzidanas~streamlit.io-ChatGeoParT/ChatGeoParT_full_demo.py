import os
import sys
import time
import uuid
import math
import openai
import tiktoken
import streamlit as st
import reveal_slides as rs
import streamlit_tags as stt

from io import BytesIO
from zipfile import ZipFile
from code_editor import code_editor

num_intro_slides = 3

if 'markdown' not in st.session_state:
    st.session_state.markdown = ""
if 'reveal' not in st.session_state:
    st.session_state.reveal = "new-start"
if 'tokens' not in st.session_state:
    st.session_state.tokens = 0
if 'delta' not in st.session_state:
    st.session_state.delta = 0
if 'lastdelta' not in st.session_state:
    st.session_state.lastdelta = 0
if 'answerfiletxt' not in st.session_state:
    st.session_state.answerfiletxt = ""

## OpenAI API Credentials setup ====================================================

# Openai local configuration

with st.sidebar:
    st.markdown("## Configuration")
    with st.expander("OpenAI API Credentials"):
        default_openai_key = ""
        default_openai_org = ""
        if openai.api_key and openai.organization:
            default_openai_key = openai.api_key
            default_openai_org = openai.organization
        openai_key = st.text_input("Key", type="password", value=default_openai_key)
        openai_org = st.text_input("Organization", type="password", value=default_openai_org)
        if openai_org and openai_key:
            openai.organization = openai_org
            openai.api_key = openai_key
            try:
                openai.Model.list()
                st.success("Success!! API credentials registered.")
            except:
                st.error("Invalid API credentials!")

if openai_key == "" or openai_org == "":
    st.warning(":arrow_left: Please provide an OpenAI API key and organization in the sidebar.")

## App setup ========================================================================
with st.sidebar:
    with st.expander("Settings"):
        request_delay = st.number_input("Delay between API requests to avoid rate limit (seconds)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        multiplier = st.number_input("Multiplier to increase values on game board", min_value=1, max_value=8, value=1, step=1)
        question_timer = st.number_input("Time given to players to answer (seconds)", min_value=0, max_value=60, value=10, step=1)

## Functions ========================================================================
def create_download_zip_file(markdown_file_string, css_file_string, answer_file_string, app_file_string):
    """
    returns: zip archive
    """
    archive = BytesIO()

    with ZipFile(archive, 'w') as zip_archive:
        # Create three files on zip archive
        with zip_archive.open('ChatGeoParT/style.css', 'w') as file1:
            file1.write(css_file_string.encode())
        
        with zip_archive.open('ChatGeoParT/games/game.md', 'w') as file2:
            file2.write(markdown_file_string.encode())

        with zip_archive.open('ChatGeoParT/games/answers.txt', 'w') as file3:
            file3.write(answer_file_string.encode())

        with zip_archive.open('ChatGeoParT/ChatGeoParT.py', 'w') as file4:
            file4.write(app_file_string.encode())

    return archive

token_count_at_start = 0
def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

messages=[
          {"role": "system", "content": "You are a helpful trivia game creation assistant."}
         ]
def query_ai(message_text, model="gpt-3.5-turbo", temperature=0.7):
    number_of_tokens = st.session_state.tokens + num_tokens_from_string(message_text)
    messages.append({"role": "user", "content": message_text})
    try:
        output = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    )
    except openai.error.Timeout as e:
        #Handle timeout error, e.g. retry or log
        st.error(f"ERROR! OpenAI API request timed out: {e}")
        pass
    except openai.error.APIError as e:
        #Handle API error, e.g. retry or log
        st.error(f"ERROR! OpenAI API returned an API Error: {e}")
        pass
    except openai.error.APIConnectionError as e:
        #Handle connection error, e.g. check network or log
        st.error(f"ERROR! OpenAI API request failed to connect: {e}")
        pass
    except openai.error.InvalidRequestError as e:
        #Handle invalid request error, e.g. validate parameters or log
        st.error(f"ERROR! OpenAI API request was invalid: {e}")
        pass
    except openai.error.AuthenticationError as e:
        #Handle authentication error, e.g. check credentials or log
        st.error(f"ERROR! OpenAI API request was not authorized: {e}")
        pass
    except openai.error.PermissionError as e:
        #Handle permission error, e.g. check scope or log
        st.error(f"ERROR! OpenAI API request was not permitted: {e}")
        pass
    except openai.error.RateLimitError as e:
        #Handle rate limit error, e.g. wait or log
        st.error(f"ERROR! OpenAI API request exceeded rate limit. Please wait a few seconds and try again.")
        pass
    except Exception as e:
        #Handle other exceptions, e.g. log
        st.error(f"ERROR! OpenAI API request failed: {e}")
        pass

    if output.choices[0].message.content:
        st.session_state.tokens = number_of_tokens + num_tokens_from_string(output.choices[0].message.content)
        return output.choices[0].message.content
    else:
        return "Error: No response from AI."

def get_jeopardy_trivia(category):
    response = query_ai("The Jeopardy category is '" + category + "' and the dollar amounts under this category are '$200', '$400', '$600', '$800', '$1000' in order of increasing difficulty. Generate a jeopardy style question + answer for each dollar amount in the category. Provide a list of the questions each with their corresponding answer in the format '- dollar amount | Question : Answer\n'.")

    trivia_questions = [[0],[0],[0],[0],[0]]
    if response:
        lines = response.split("\n")
        for line in lines:
            if line.startswith("$") or line.startswith("-"):
                row = line.split("|")
                if len(row) == 2:
                    qa = row[1].replace("Question:", "").replace("question:", "").replace("q:", "").replace("Q:", "").split(":")
                    if len(qa) == 2:
                        amount = row[0].replace("$", "").replace("-", "")
                        question = qa[0].replace("answer", "").replace("Answer", "").strip()
                        answer = qa[1].replace("answer", "").replace("Answer", "").strip()
                        if "200" in amount:
                            trivia_questions[0] = format_qa_response([question, answer])
                        elif "400" in amount:
                            trivia_questions[1] = format_qa_response([question, answer])
                        elif "600" in amount:
                            trivia_questions[2] = format_qa_response([question, answer])
                        elif "800" in amount:
                            trivia_questions[3] = format_qa_response([question, answer])
                        elif "1000" in amount:
                            trivia_questions[4] = format_qa_response([question, answer])
                
    return trivia_questions

def retry_get_jeopardy_trivia(category, number_of_tries=2):
    trivia_questions = [[0],[0],[0],[0],[0]]
    for i in range(number_of_tries):
        time.sleep(request_delay)
        trivia_questions = get_jeopardy_trivia(category)
        check = True
        for array in trivia_questions:
            if len(array) < 2 or array[0] == 0:
                check = False
        if check:
            return trivia_questions
    return trivia_questions

def format_qa_response(response):
    question = response[0]
    answer = response[1]
    six_ws = ["who", "what", "when", "where", "why", "which"]
    etre = ["is", "are", "was", "were"]
    words_in_q = question.replace("-", "").replace(":", "").strip().split(" ")
    ans_start = ""
    if words_in_q[0].lower() in six_ws and words_in_q[1].lower() in etre:
        question = question.replace(words_in_q[0], "", 1).replace(words_in_q[1], "", 1).replace(words_in_q[2], words_in_q[2].capitalize(), 1).strip()
        ans_start = words_in_q[0].capitalize() + " " + words_in_q[1].lower() + " "
    elif words_in_q[0].lower() in six_ws:
        if words_in_q[0].lower() == "who":
            question = question.replace(words_in_q[0], "This person", 1)
            ans_start = "Who is "
        elif words_in_q[0].lower() == "what":
            question = question.replace(words_in_q[0], "This", 1)
            ans_start = "What is "
    words_in_a = answer.replace("-", "").replace(":", "").strip().split(" ")
    if words_in_a[0].lower() not in six_ws:
        answer = ans_start + answer.replace(words_in_a[0], words_in_a[0].lower(), 1).strip()
    if question.strip().endswith("?"):
        question = question.strip().rsplit("?", 1)[0]
    answer = answer.strip()
    if not answer.endswith("?"):
        if answer.endswith("."):
            answer = answer.rsplit(".", 1)[0]
        answer = answer.strip() + "?"

    return [question, answer.upper()]


## Main App ========================================================================
slide_markdown = r"""<section data-markdown="" data-separator-vertical="^--$" data-separator-notes="^Answer:" >
<script type="text/template">
## [Welcome to Chat GeoParT!](#/1)"""

slide_markdown_close = r"""
</script>
</section>
<script type="application/javascript">
    function findLink(el) {
        if (el.tagName == 'A' && el.href) {
            return el;
        } else if (el.parentElement) {
            return findLink(el.parentElement);
        } else {
            return null;
        }
    };

    function callback(e) {
        const link = findLink(e.target);
        if (link == null) { return; }
        e.preventDefault();
        // Do stuff here
        link.classList.add("clicked");
    };

    document.addEventListener('click', callback, false);
</script>
"""

st.title("Chat GeoParT!")

with st.expander("Instructions and tips"):
    st.markdown("""**This app lets you generate a Jeopardy style trivia game with questions and answers.** 
- _Start by entering 6 topics of your choice to be used as the categories._
- _After all 6 categories are entered, the New Game button will be enabled. Click it to generate the game (this might take a couple of minuites)._
- _The game will be generated as a series of slides which you can navigate through by clicking text that is **yellow**_
- _For the full screen experience, make sure to give focus to the slides by clicking anywhere on a slide and then press `F` on you keyboard._
- _You can also use the arrow keys to navigate through the slides in linear order (not in gameplay order)._
- _You can increase the amounts by changing the multiplier setting in the sidebar._
- _Check out the settings in the sidebar for more options._
- _After the slides have been generated, you can download the them along with the answers and a simple streamlit app that allows you to play the Chat GeoParT games. Run this app as you would any streamlit app and you will be able to play any of the game files stored in the 'game' folder._""")

categories = stt.st_tags(label="Enter 6 trivia categories", suggestions="Enter a category", maxtags=6, key="categories")
cola, colb, colc = st.columns([1.1,4.6,1])
if categories:
    if cola.button("New Game", disabled=len(categories) < 6):
        token_count_at_start = st.session_state.tokens
        with st.spinner("Generating ..."):
            intro_offset = num_intro_slides
            slide_markdown += "\n---\n"+ r"""<!-- .slide: data-transition="zoom" -->""" + "\n ## [The categories are ... ](#/2) "
            for category in categories:
                slide_markdown += f"\n---\n"+ r"""<!-- .slide: data-transition="zoom" -->""" + f"\n ## [{category.upper()}](#/{intro_offset})"
                intro_offset += 1
            slide_markdown += "\n---\n" + r"""<!-- .slide: data-transition="fade" data-background-image="https://cdn.vox-cdn.com/thumbor/wEcBsqpKaKmrw6TWYNIDQfOPENk=/172x118:2400x1232/fit-in/1200x600/cdn.vox-cdn.com/uploads/chorus_asset/file/19577016/jeopardy_02.jpg" data-background-size="118%" data-background-position="20%" -->""" + "\n"
            slide_markdown += "|"
            jeopardy_set = []
            loops_to_time = 1
            loop_count = 1
            timer_start = time.time()
            for category in categories:
                slide_markdown += f" {category.upper()} |"
                qa_array = retry_get_jeopardy_trivia(category) 

                if qa_array[0][0] != 0:
                    jeopardy_set.append(qa_array)
                
                time_taken = time.time() - timer_start
                if loop_count < 2 and time_taken > 10:
                    colb.write("This might take a few minutes ...")
                
                loop_count += 1

            slide_markdown += "\n|:-:|:-:|:-:|:-:|:-:|:-:|"
            for row_index in range(5):
                slide_markdown += "\n|"
                for column_index in range(6):
                    slide_markdown += f"[${(row_index+1)*200*multiplier}](#/{column_index*5 + row_index + intro_offset})|"
            slide_markdown += "\n"
            if len(jeopardy_set) == 6:
                answerfiletxt = ""
                for column in range(6):
                    for row in range(5):
                        slide_markdown += "\n---\n"
                        slide_markdown += f'<!-- .slide: data-transition="zoom" -->\n### ({categories[column].upper()}) \n# [${((row+1)*200*multiplier)}](#/{column*5 + row + intro_offset}/1)'
                        slide_markdown += "\n--\n"
                        slide_markdown += f'<!-- .slide: data-transition="zoom" data-autoslide="{question_timer*1000}" -->\n### [{jeopardy_set[column][row][0]}](#/{column*5 + row + intro_offset}/2)\nAnswer:{jeopardy_set[column][row][1]}'
                        slide_markdown += "\n--\n"
                        slide_markdown += f'<!-- .slide: data-transition="zoom-in fade-out" -->\n### [{jeopardy_set[column][row][1]}](#/{intro_offset - 1})'
                        answerfiletxt += f"{categories[column].upper()}\n\t[{((row+1)*200*multiplier)}]: Q:{jeopardy_set[column][row][0]}\n\t\tA:{jeopardy_set[column][row][1]}\n\n"
                slide_markdown += slide_markdown_close

                new_key = str(uuid.uuid4())
                st.session_state.reveal = new_key
                st.session_state.markdown = slide_markdown
                st.session_state.lastdelta = st.session_state.delta
                st.session_state.delta = st.session_state.tokens - token_count_at_start 
                st.session_state.answerfiletxt = answerfiletxt

## Presenting generated content and data ======================================================
btn_settings_editor_btns = [{
                                "name": "copy",
                                "feather": "Copy",
                                "hasText": True,
                                "alwaysOn": True,
                                "commands": ["copyAll"],
                                "style": {"top": "0rem", "right": "0.4rem"}
                            },
                            {
                                "name": "update",
                                "feather": "RefreshCw",
                                "primary": True,
                                "hasText": True,
                                "showWithIcon": True,
                                "commands": ["submit"],
                                "style": {"bottom": "0rem", "right": "0.4rem"}
                            }]

reveal_snippets = [[{ 
                        "name": '--', 
                        "code": '\n---\n## New Horizontal Slide\n' 
                    },
                    { 
                        "name": 'new slide (horizontal)', 
                        "code": '\n---\n## New Horizontal Slide\n' 
                    },
                    { 
                        "name": '-', 
                        "code": '\n--\n## New Vertical Slide\n' 
                    },
                    { 
                        "name": 'new slide (vertical)', 
                        "code": '--\n## New Vertical Slide\n' 
                    }, 
                    { 
                        "name": '<!-- .s', 
                        "code": '<!-- .slide:  -->'
                    },
                    { 
                        "name": '<!-- .e', 
                        "code": '<!-- .element: class="fragment" class="fragment" -->'
                    },
                    {
                        "name": 'slide attributes', 
                        "code": '<!-- .slide:  -->'
                    }, 
                    { 
                        "name": 'element attributes', 
                        "code": '<!-- .element: class="fragment" class="fragment" -->'
                    } 
                  ],""]

if 'css' not in st.session_state:
    st.session_state.css = """
body.reveal-viewport {
    background: #2b09cf;             /*slide background color*/
}
.reveal table {
    background: #1f069d;             /*Jeopardy board background color*/
}
.reveal table thead {                /*top|category row*/
    border: 14px solid #000000;  
}
.reveal table tr {                   /*all rows below top*/
    border: 11px solid #000000; 
}
.reveal table th, .reveal table td { /*every element of table*/
    width: 11rem;
    height: 9.5rem;
    border: 11px solid #000000;
}
.reveal table th {                   /*category element of table*/
    color: white;
    font-weight: bold;
    font-size: 0.6em;
    height: 10rem;
    text-shadow: 4px 4px #000000;
}
.reveal table td {                   /*non-category elements of table*/
    color: #ffc69675;
    font-weight: bold;
    font-size: 1.65em;
}
.reveal table td a {                 /*links in table*/
    text-shadow: 4px 4px #000000;
}
.reveal section > h1 a {             /*header 1 links outside table*/
    font-size: 2.5em; 
}
.reveal section > h2 a {             /*header 2 links outside table*/
    font-size: 1.75em;
}
.reveal section > h3 a {             /*header 3 links outside table*/
    font-size: 1.55em;
}

.reveal table td a.clicked {         /*links in table after clicked*/
    pointer-events: none; 
    cursor: default;
    color: #ffc69675;
    text-shadow: none;
}
"""

if st.session_state.markdown != "":  
    with st.expander("Customize"):
        markup_response_dict = code_editor(st.session_state.markdown, lang="markdown", height=25, snippets=reveal_snippets, buttons=btn_settings_editor_btns, allow_reset=True, key="code_editor")
        if markup_response_dict['type'] == "submit" and len(markup_response_dict['text']) != 0:
            st.session_state.markdown = markup_response_dict['text']
        
        css_response_dict = code_editor(st.session_state.css, lang="css", height=15, buttons=btn_settings_editor_btns, allow_reset=True, key="css_editor")
        if css_response_dict['type'] == "submit" and len(css_response_dict['text']) != 0:
            st.session_state.css = css_response_dict['text']

    reveal_state = rs.slides(st.session_state.markdown, 
          height=410,
          config={
                  "width": 1800, 
                  "height": 1000, 
                  "minScale": 0.1,
                  "center": True, 
                  "maxScale": 4, 
                  "margin": 0.09,
                  "controls": False,
                  "history": True,
                  "plugins": ["markdown", "highlight", "katex", "notes", "search", "zoom"]
                 }, 
          theme="night",
          css=st.session_state.css,
          allow_unsafe_html=True,
          key=st.session_state.reveal
          )
    with st.expander("API Token Metrics"):
        col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 3, 1, 3, 1, 3, 2])
        col2.metric(label="Usage", value=str(round(100*st.session_state.delta/4096, 1)) + "%", delta=str(round(100*(st.session_state.delta - st.session_state.lastdelta)/4096, 1)) + "%")
        col4.metric(label="Tokens", value=st.session_state.tokens, delta=st.session_state.delta)
        col6.metric(label="Cost", value="$" + str(round((st.session_state.tokens*0.005/2500), 3)), delta="$" + str(round((st.session_state.delta * 0.005/2500), 3)))

    with open('ChatGeoParT.py', 'r') as f:
        app_to_download = f.read()

    download_file = create_download_zip_file(st.session_state.markdown, st.session_state.css, st.session_state.answerfiletxt, app_to_download)
    with open('ChatGeoParT.zip', 'wb') as f:
        f.write(download_file.getbuffer())
    download_file.close()

    with open('ChatGeoParT.zip', 'rb') as f:
        colc.download_button('Download', f, file_name='ChatGeoParT.zip')
    
