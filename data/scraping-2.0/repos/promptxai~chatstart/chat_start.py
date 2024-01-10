import streamlit as st
import os

from chatux import session, content
from genapi import model, integrate

# Setup session variables
state = st.session_state

# Integrations with LLMs, APIs, and Content sources
if 'stability' not in state:
    state.stability = session.Stability()
if 'open_ai' not in state:
    state.open_ai = session.OpenAI()
if 'google' not in state:
    state.google = session.Google()
if 'content' not in state:
    state.content = session.Content()

# User Experience
if 'ux' not in state:
    state.ux = session.UX()
if 'chat' not in state:
    state.chat = session.Chat()

# Set page config
st.set_page_config(
    page_title="ChatStart - Create, explore, generate code for a Chatbot. Fast!",
    page_icon="media/chatstart_icon_32.png",
    layout="centered",
    initial_sidebar_state=state.ux.sidebar)

# Setup API keys from environment variables or session state
GOOGLE_DEVELOPER_KEY = os.environ.get('GOOGLE_DEVELOPER_KEY', state.google.key)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', state.open_ai.key)
STABILITY_KEY = os.environ.get('STABILITY_KEY', state.stability.key)

# Replace with Stability Host if different
STABILITY_HOST=os.environ.get('STABILITY_HOST', 'grpc.stability.ai:443')
# Replace with your own custom search engine ID or use ours
CUSTOM_SEARCH_ENGINE_ID = os.environ.get('CUSTOM_SEARCH_ENGINE_ID', 'a65812bec92ed4b8c')

# Other keys are optional
if OPENAI_API_KEY:
    open_ai = model.OpenAI(OPENAI_API_KEY)
    state.ux.keys_saved = True

# Hydrate ideas
open_ai_ideas = {}
content.hydrate_ideas(category='open_ai', ideas=open_ai_ideas)

sd_ideas = {}
content.hydrate_ideas(category='sd', ideas=sd_ideas)

google_ideas = {}
content.hydrate_ideas(category='google', ideas=google_ideas)

ideas = open_ai_ideas | sd_ideas if STABILITY_KEY else open_ai_ideas
ideas = ideas | google_ideas if GOOGLE_DEVELOPER_KEY else ideas

# Hydrate parameters
state.ux.parameters = content.hydrate_parameters()

stability = None
if STABILITY_KEY:
    stability = model.Stability(STABILITY_KEY)
    state.stability.api = stability.api

def generate_code():
    st.markdown('### Add {idea} to your app'.format(idea = state.chat.idea))
    st.markdown('**Step 1:**' + ' ' + 'Use the following code for ChatGPT API call.')
    st.markdown(
        '''```python
import openai

openai.ChatCompletion.create(
        model="{model}",
        messages= # Step 2: Copy the messages list here
        max_tokens=100,
        temperature={temperature},)
)
        '''.format(model=state.open_ai.model, temperature=state.chat.temperature))
    st.markdown('**Step 2:**' + ' ' + 'Copy the following messages list and assign to `messages` variable.')
    st.code(state.chat.messages)

def setup_conversation():
    last_user = ideas[state.chat.idea].rfind('User: ')
    state.chat.conversation = '\n'.join(ideas[state.chat.idea].splitlines()[:-last_user])
    state.chat.prompt = ideas[state.chat.idea][last_user + 6:]
    state.chat.conversation = ideas[state.chat.idea][:last_user]

    state.chat.temperature = state.ux.parameters[state.chat.idea]['temperature']
    state.ux.code = False

st.sidebar.markdown("### 💡 Select Idea")
# create a form to collect user input
with st.sidebar.form(key="idea_form"):
    # add a text field to the form
    state.chat.idea = st.selectbox("Type or scroll", ideas.keys())
    # add a submit button to the form
    submit_button = st.form_submit_button(label="Explore Chat", disabled=state.ux.keys_saved is False)

    # if the form is submitted
    if submit_button:
        setup_conversation()

st.sidebar.markdown("### 🧠 Change Model")

with st.sidebar.form(key="model_form"):
    state.open_ai.model = st.selectbox("OpenAI Model", 
        options=['gpt-3.5-turbo', 'gpt-4'])
    submit_button = st.form_submit_button(label="Change Model", disabled=state.ux.keys_saved is False)

    if submit_button:
        st.experimental_rerun()

if state.ux.code:
    st.sidebar.markdown("### 🪄 Get code")
    st.sidebar.button('Generate tutorial with code', on_click=generate_code)

st.sidebar.markdown('### 🔒 User Account')
if state.ux.keys_saved is False:
    with st.sidebar.form(key="key_form"):
        state.open_ai.key = st.text_input("OpenAI Key (Required)", 
            value=state.open_ai.key, type="password", 
            help="Get your required OpenAI Key from [here](https://beta.openai.com/account/api-keys).")
        state.google.key = st.text_input("Google Key", 
            value=state.google.key, type="password", 
            help="Get your optional Google Key from [here](https://console.cloud.google.com/apis/credentials).")
        state.stability.key = st.text_input("Stability Key", 
            value=state.stability.key, type="password",
            help="Get your optional Stability Key from [here](https://beta.dreamstudio.ai/membership?tab=apiKeys).")
        submit_keys = st.form_submit_button(label="Activate Session")
        if submit_keys:
            if state.open_ai.key == '':
                st.sidebar.error('OpenAI Key is required.')
            else:
                state.ux.keys_saved = True
                st.experimental_rerun()

if state.ux.keys_saved is True:
    st.sidebar.markdown('**OpenAI Model:** ' + state.open_ai.model)
    st.sidebar.markdown('**ChatGPT Tokens Used:** ' + str(state.open_ai.tokens))
    st.sidebar.markdown('**ChatGPT Runs:** ' + str(state.open_ai.chatgpt_runs))
    st.sidebar.markdown('**Stability Runs:** ' + str(state.stability.runs))
    st.sidebar.markdown('**DALL.E Runs:** ' + str(state.open_ai.dalle_runs))
    st.sidebar.markdown('**Google API Runs:** ' + str(state.google.runs))

logo_nav1, logo_nav2 = st.columns([3, 5])
with logo_nav1:
    st.image('media/chatstart_logo_wide_w250.png', width=250)
with logo_nav2:
    st.markdown("#### " + state.chat.idea if state.chat.conversation else "")
if state.ux.parameters[state.chat.idea]['model'] == 'gpt-4' and state.open_ai.model != 'gpt-4':
    st.markdown('*GPT-4 recommended for this idea*')
else:
    st.markdown("**Create, Explore, and Generate Chatbots. Fast!**")

if not state.chat.conversation:
    content.intro()

if state.chat.conversation:
    # Render chat conversation
    state.ux.icon = state.chat.idea.split()[0]
    st.markdown(state.chat.conversation
                .replace('System:', '⚙️ &nbsp;&nbsp;')
                .replace('User:', '\n👤 &nbsp;&nbsp;')
                .replace('Assistant:', '\n' + state.ux.icon + ' &nbsp;&nbsp;'))
    
    # Apply integrations
    state.chat.integrate = False
    if '"' in state.chat.conversation and 'Molecule Generator' in state.chat.idea:
        integrate.molecule(state.chat.conversation)
        state.chat.integrate = True

    if '|' in state.chat.conversation and 'Molecule Generator' in state.chat.idea:
        state.dalle_image = integrate.dalle(conversation=state.chat.conversation, model=open_ai, separator='|')
        state.open_ai.dalle_runs += 1
        st.image(state.dalle_image, caption='DALL.E Generated Image')
        state.chat.integrate = True

    if '"' in state.chat.conversation and 'DALL.E Expert Artist' in state.chat.idea:
        state.dalle_image = integrate.dalle(state.chat.conversation, open_ai)
        state.open_ai.dalle_runs += 1
        st.image(state.dalle_image, caption='DALL.E Generated Image')
        state.chat.integrate = True

    if '"' in state.chat.conversation and 'Shopping Recommender' in state.chat.idea:
        res = integrate.google_image(state.chat.conversation, num=3,
            key=GOOGLE_DEVELOPER_KEY, cx=CUSTOM_SEARCH_ENGINE_ID)
        state.google.runs += 1
        content.render_carousel(res, num=3)
        state.chat.integrate = True

    if '"' in state.chat.conversation and 'Stable Diffusion Story Generator' in state.chat.idea:
        state.stability.image = integrate.stability(state.chat.conversation, stability)
        state.stability.runs += 1
        st.image(state.stability.image, caption='Stable Diffusion Generated Image')
        state.chat.integrate = True
        
    if 'Dataset Generator' in state.chat.idea and '```' in state.chat.conversation:
        state.content.dataframe = integrate.dataframe(state.chat.conversation)
        st.dataframe(state.content.dataframe)
        state.chat.integrate = True
    
    if 'Vegalite Chart Generator' in state.chat.idea and '```' in state.chat.conversation:
        state.content.vegalite = integrate.vegalite(state.chat.conversation)
        st.vega_lite_chart(state.content.vegalite, use_container_width=True)
        state.chat.integrate = True

    if 'youtube.com' in state.chat.conversation:
        state.content.youtube = integrate.youtube(state.chat.conversation)
        st.video(state.content.youtube)
        state.chat.integrate = True        

if state.chat.conversation:
    with st.form(key="chat_form"):
        c1, c2 = st.columns([9, 1])
        with c1:
            user_input = st.text_area("👤 &nbsp;&nbsp;Your message here", value=state.chat.prompt)
        with c2:
            st.caption('Discuss')
            submit_button = st.form_submit_button(label=state.ux.icon)
        if submit_button:
            state.chat.conversation += '\nUser: ' + user_input
            state.chat.prompt = ''
            # parse the state.chat.conversation into messages list considering multi-line User, System, and Assistant messages.
            # If new line does not have a role, it is considered as continuation of current line.
            state.chat.messages = []
            for line in state.chat.conversation.splitlines():
                if line.startswith('User:'):
                    state.chat.messages.append({"role": "user", "content": line[5:]})
                elif line.startswith('System:'):
                    state.chat.messages.append({"role": "system", "content": line[7:]})
                elif line.startswith('Assistant:'):
                    state.chat.messages.append({"role": "assistant", "content": line[10:]})
                else:
                    state.chat.messages[-1]["content"] += '\n' + line
            
            # call openai api
            response = open_ai.chat(
                model=state.open_ai.model,
                messages=state.chat.messages,
                max_tokens=1000,
                temperature=state.chat.temperature)

            state.open_ai.chatgpt_runs += 1
            state.open_ai.tokens += response.usage.total_tokens
            state.chat.conversation += '\n' + 'Assistant: ' + response.choices[0].message.content
            state.ux.code = True
            st.experimental_rerun()

