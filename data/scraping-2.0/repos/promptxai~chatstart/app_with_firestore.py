import streamlit as st
import openai
import pandas as pd
import json
import io
from google.cloud import firestore
import datetime
from google.oauth2 import service_account
import os
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from googleapiclient.discovery import build

from chatstart_sdk import session

GOOGLE_DEVELOPER_KEY = os.environ['GOOGLE_DEVELOPER_KEY']
FIREBASE_PROJECT_ID = os.environ['FIREBASE_PROJECT_ID']

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
if 'user' not in state:
    state.user = session.User()
if 'ux' not in state:
    state.ux = session.UX()
if 'chat' not in state:
    state.chat = session.Chat()

# Set page config
st.set_page_config(
    page_title="ChatStart - Ideate, explore, generate code for ChatGPT integration with your app",
    page_icon="chatstart_icon_32.png",
    layout="centered",
    initial_sidebar_state=state.ux.sidebar)

# Setup ideas
ideas = {
"🎨 DALL.E Expert Artist":
'''System: You are a DALL.E Expert Artist who will ask the user questions about various 
features of art they want to create and then generate a prompt using 
DALL.E advanced features to get the best results.
User: I want to create an image of a futuristic city.
Assistant: Do you have a reference movie, style, or concept this is based on?
User: Yes, it is based on the movie Blade Runner.''',

"💻 Mac Terminal": 
'''System: You are a Mac Terminal. Based on natural language instructions respond
only in single line shell commands. If the command is not recognized, respond with "Command not found".
User: List files of current directory and copy to a text file called files_list.txt
Assistant: `ls > files_list.txt`
User: Parse paragraphs from https://en.wikipedia.org/wiki/Earth into a text file called earth.txt
''',        

"🛍️ Shopping Recommender":
'''System: You are a Shopping Recommender. Ask the user questions about their preferences,
tasters, and lifestyle. Respond with the brand, name, and model of the product enclosed in double quotes.
Also explain why the product is recommended in a crisp single sentence.
User: I need some shopping recommendations.
Assistant: Tell me more about your tastes and lifestyle, so I can recommend the best products.
User: I am an artist, avid gamer, and an audiophile. I live in an ultra-modern loft apartment.
Assistant: Nice! What are you interested in buying?
User: I am looking for a Smart TV.
''',

"🍿 Movie Database": 
'''System: You are a Movie Database that responds with movies related information.
Provide information in a crisp single sentence.
User: What is the movie rating of The Matrix?
Assistant: The Matrix is rated 8.7/10 on IMDb.
User: What is the link of the official trailer?''',

"👩‍⚕️ Doctor": 
'''System: You are a Doctor who understands symptoms by asking follow up questions 
and when sufficient symptoms are known provides a diagnosis.
User: I have a headache.
Assistant: How long have you had a headache?
User: For a week.
Assistant: Do you have blurred vision?
User: No, but I have a fever.''',

"💊 Pharmacist": 
'''System: You are a Pharmacist who provides information about medicines, their usual dosage, 
side effects, and interactions. Provide information in a crisp single sentence.
User: What is the usual dosage of paracetamol?
Assistant: The usual dosage of paracetamol is 1-2 tablets every 4-6 hours.
User: What are the side effects of paracetamol?''',

"🧚 Stable Diffusion Story Generator":
'''System: You are a Stable Diffusion Story Generator who creates 
Stable Diffusion prompts in double quotes, which represent crisp one sentence 
story pregressions based on user interactions.
User: I am in the mood for some science fiction.
Assistant: Ok, how do you want to start the story?
User: Spaceship Anubis begins mission to explore a distant unknown galaxy.''',

"📊 Vegalite Chart Generator":
'''System: You are a Vegalite Chart Generator that can also generate datasets to use in the chart.
You ask the user a few questions about the data they want to visualize, type of chart,
style, and other options. Then generate a vegalite chart code based on the instructions.
User: Create a chart.
Assistant: What data source should I use?
User: Create a list of 10 most populous cities in the world.''',

"🗂️ Dataset Generator":
'''System: You are a Dataset Generator. Create a code fenced csv based on user instructions.
User: Create a dataset of tallest buildings in the world.'''
}

def init_stability_api():
    state.stability.api = client.StabilityInference(
        key=os.environ['STABILITY_KEY'], # API Key reference.
        verbose=True, # Print debug messages.
        engine="stable-diffusion-v1-5", # Set the engine to use for generation. 
        # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0 
        # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
    )
    state.stability.initialized = True

def generate_art_sd(prompt, size=512) -> Image:
    if state.stability.initialized is False:
        init_stability_api()

    answers = state.stability.api.generate(
        prompt=prompt,
        cfg_scale=8.0,
        width=size, # Generation width, defaults to 512 if not included.
        height=size, # Generation height, defaults to 512 if not included.
        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
        # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
        # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, 
        # k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
    )

    state.stability.runs += 1

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, save generated images.
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                return img


def generate_code():
    st.markdown('### Add {idea} to your app'.format(idea = state.chat.idea))
    st.markdown('**Step 1:**' + ' ' + 'Use the following code for ChatGPT API call.')
    st.markdown(
        '''```python
import openai

openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= # Step 2: Copy the messages list here
        max_tokens=100,
        temperature=0.2,)
)
        ''')
    st.markdown('**Step 2:**' + ' ' + 'Copy the following messages list and assign to `messages` variable.')
    st.code(state.chat.messages)

if not state.user.waitlisted and not state.user.login_form and not state.user.authenticated:
    with st.sidebar.form('waitlist_form'):
        st.markdown('### Waitlist for ChatStart')
        email = st.text_input("Email")
        code = st.text_input("Code")
        st.caption("If you have received a license key or promo code, enter it here.")
        submit_button = st.form_submit_button(label="Apply")

        if submit_button:
            key_dict = json.loads(st.secrets["textkey"])
            creds = service_account.Credentials.from_service_account_info(key_dict)
            db = firestore.Client(credentials=creds, project=FIREBASE_PROJECT_ID)
            doc_ref = db.collection("waitlist").document(email)
            doc = doc_ref.get()
            if doc.exists:
                st.sidebar.error("Email already in waitlist")
            else:
                doc_ref.set({
                    'code': code,
                    'timestamp': datetime.datetime.now()
                })
                state.user.waitlisted = True
                st.sidebar.success("You have been added to the waitlist. We will notify you when you can use ChatStart.")

def login():
    state.user.login_form = True

if not state.user.authenticated and not state.user.login_form:
    st.sidebar.button('Login', on_click=login)

if not state.user.authenticated and state.user.login_form:
    with st.sidebar.form(key="login_form"):
        login = st.text_input("Login")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label="Login")

        if submit_button:
            key_dict = json.loads(st.secrets["textkey"])
            creds = service_account.Credentials.from_service_account_info(key_dict)
            db = firestore.Client(credentials=creds, project="chatstart-aa174")
            doc_ref = db.collection("users").document(login)
            doc = doc_ref.get()
            if doc.exists:
                if doc.to_dict()['password'] == password:
                    state.user.authenticated = True
                    state.user.login = login
                    state.user.login_form = False
                    st.sidebar.success("Login successful")
                else:
                    st.sidebar.error("Incorrect password")
            else:
                st.sidebar.error("User does not exist")

st.sidebar.markdown("### 💡 Select Idea")
# create a form to collect user input
with st.sidebar.form(key="role_form"):
    # add a text field to the form
    state.chat.idea = st.selectbox("Start typing to search or select", ideas.keys())
    # add a submit button to the form
    submit_button = st.form_submit_button(label="Explore Chat", disabled=not state.user.authenticated)

    # if the form is submitted
    if submit_button:
        # set the conversation to the selected idea after removing the last line and joining the lines with new line
        state.chat.conversation = '\n'.join(ideas[state.chat.idea].splitlines()[:-1])
        # set the last line in ideas as the user prompt
        state.chat.prompt = ideas[state.chat.idea].splitlines()[-1].replace('User: ', '')
        state.ux.code = False

st.sidebar.markdown('### 🔒 User Account')
if state.user.authenticated:
    # st.sidebar.markdown('Welcome ' + state.user.login)
    st.sidebar.markdown('**ChatGPT Tokens Used:** ' + str(state.open_ai.tokens))
    st.sidebar.markdown('**ChatGPT Runs:** ' + str(state.open_ai.chatgpt_runs))
    st.sidebar.markdown('**Stability Runs:** ' + str(state.stability.runs))
    st.sidebar.markdown('**DALL.E Runs:** ' + str(state.open_ai.dalle_runs))
    st.sidebar.markdown('**Google API Runs:** ' + str(state.google.runs))

nav1, nav2 = st.columns([7, 1])
with nav1:
    st.write('Please waitlist for 💬&nbsp; ChatStart to enable featues.')
with nav2:
    if st.button('Waitlist'):
        state.ux.sidebar = 'expanded'
        st.experimental_rerun()

logo_nav1, logo_nav2 = st.columns([3, 5])
with logo_nav1:
    st.image('chatstart_logo_wide_w250.png', width=250)
with logo_nav2:
    st.markdown("#### " + state.chat.idea if state.chat.conversation else "")

st.markdown("**Ideate, explore, generate code for ChatGPT integration with your app**")

if not state.chat.conversation:
    st.markdown('### ChatGPT and generative AI models are about to transform every industry')

    st.video('walkthrough.webm')

    st.markdown('#### 💬&nbsp; ChatStart helps stay ahead of the curve in three easy steps')
    st.markdown('### 1. Select an idea')
    st.success('''Start by selecting an idea for your app, startup, or business project.
    We will continue to add more ideas to the list covering industries, roles, and use cases.
    &nbsp;💬&nbsp; ChatStart Premium users can also save their own ideas privately or share with others.''')
    st.image('ideate.png', width=350)
    
    st.markdown('### 2. Explore in chat')
    st.info('''Once you have selected an idea, you can explore it in a custom chat powered by ChatGPT
    and other generative AI models. You can fine tune your chatbot by simply chatting with it.
    &nbsp;💬&nbsp; ChatStart Premium users can also save their chat explorations privately or share with others.''')

    st.image('explore.png')

    st.markdown('### 3. Generate tutorial and code')
    st.warning('''Once satisfied with your chat exploration, you can hit `Generate Code` button. This will generate
    a custom tutorial for this idea and code for integrating ChatGPT with your app. Your entire chat exploration
    will be available to fine tune your chatbot.
    &nbsp;💬&nbsp; ChatStart Premium users can also get access to advanced code and tutorials to integrated
    with multiple APIs and models.''')

    st.image('tutorial_code.png')

    st.markdown('---')

    st.markdown('## More Features')

    st.markdown('### Contextual Image Search')
    st.success('''Integrate Google Image Search within your chat to discover images and links.
    You can use this for creating chat based visual search apps for shopping, education, etc.
    &nbsp;💬&nbsp; ChatStart Premium users can also save these discovered links and get access to advanced code
    for Google API integrations.
    ''')

    st.image('shopping.png')

    st.markdown('### Chain multiple models')
    st.info('''Make your chat sessions super productive by chaining ChatGPT text with DALL.E image generations.
    &nbsp;💬&nbsp; ChatStart Premium users can also get access to advanced code and tutorials to integrated
    with multiple APIs and models.''')

    st.image('chaining.png')

    st.markdown('### Generate Python Dataframes')
    st.success('''Integrate ChatStart generated code within your Python apps getting access to native constructs
    like Pandas DataFrames, JSON objects, data structures like lists and dictionaries directly from within
    a chat exploration.
    &nbsp;💬&nbsp; ChatStart Premium users can also save these native objects and get access to advanced code
    and tutorials to perform improved integrations.''')

    st.image('datasets.png')

    st.markdown('### Browse Media Inplace')
    st.warning('''Use chat to browse media, see images, watch related videos in place.
    &nbsp;💬&nbsp; ChatStart Premium users can save these discovered media objects
    and also connect with multiple media providers.''')

    st.image('media.png')

    st.markdown('### Query Data and Generate Charts Interactively')
    st.info('''Use natural language to query world knowledge for generating custom datasets 
    and then visualize these datasets into charts and analytics, which can be modified using plain English.
    &nbsp;💬&nbsp; ChatStart Premium users can also save these datasets and analytics for
    retrieving at a later date or sharing with others.''')

    st.image('charts.png')

    st.markdown('### Simulate Expert Advisor')
    st.image('specialist.png')

    st.markdown('### Specialized Learning Tool')
    st.image('learning.png')

    st.markdown('### Visual Story Generation')
    st.image('story.png')


if state.chat.conversation:
    # get icon from the idea name
    state.ux.icon = state.chat.idea.split()[0]
    st.markdown(state.chat.conversation
                .replace('System:', '⚙️ &nbsp;&nbsp;')
                .replace('User:', '\n👤 &nbsp;&nbsp;')
                .replace('Assistant:', '\n' + state.ux.icon + ' &nbsp;&nbsp;'))
    # if state.chat.conversation contains DALL.E prompt in a code fenced block, then use the prompt to generate a new image
    if '"' in state.chat.conversation and 'DALL.E Expert Artist' in state.chat.idea:
        num_quotes = state.chat.conversation.count('"')
        prompt = state.chat.conversation.split('"')[num_quotes - 1]
        # generate image from prompt
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        state.open_ai.dalle_runs += 1

        generated_image = response['data'][0]['url']
        st.image(generated_image, caption='DALL.E Generated Image')
        state.dalle_image = generated_image
    if '"' in state.chat.conversation and 'Shopping Recommender' in state.chat.idea:
        num_quotes = state.chat.conversation.count('"')
        search_query = state.chat.conversation.split('"')[num_quotes - 1]
        
        service = build("customsearch", "v1", developerKey=GOOGLE_DEVELOPER_KEY)

        res = (service.cse().list(
                    q=search_query,
                    cx="a65812bec92ed4b8c",
                    num=3,
                    searchType="image",
                    filter="1",
                    safe="active",
                ).execute())
        
        state.google.runs += 1

        ci1, ci2, ci3 = st.columns(3)
        with ci1:
            thumbnail = res["items"][0]["image"]["thumbnailLink"]
            link = res["items"][0]["image"]["contextLink"]
            domain = link.split('/')[2]
            st.markdown('[![]({thumbnail})]({link})'.format(thumbnail=thumbnail, link=link))
            st.markdown('[{domain}]({link})'.format(domain=domain, link=link))
        with ci2:
            thumbnail = res["items"][1]["image"]["thumbnailLink"]
            link = res["items"][1]["image"]["contextLink"]
            domain = link.split('/')[2]
            st.markdown('[![]({thumbnail})]({link})'.format(thumbnail=thumbnail, link=link))
            st.markdown('[{domain}]({link})'.format(domain=domain, link=link))
        with ci3:
            thumbnail = res["items"][2]["image"]["thumbnailLink"]
            link = res["items"][2]["image"]["contextLink"]
            domain = link.split('/')[2]
            st.markdown('[![]({thumbnail})]({link})'.format(thumbnail=thumbnail, link=link))
            st.markdown('[{domain}]({link})'.format(domain=domain, link=link))


    if '"' in state.chat.conversation and 'Stable Diffusion Story Generator' in state.chat.idea:
        num_quotes = state.chat.conversation.count('"')
        prompt = state.chat.conversation.split('"')[num_quotes - 1]
        generated_image = generate_art_sd(prompt)
        st.image(generated_image, caption='Stable Diffusion Generated Image')
        state.stability.image = generated_image

    if 'Dataset Generator' in state.chat.idea and '```' in state.chat.conversation:
        num_csv = state.chat.conversation.count('```')
        dataset_csv = state.chat.conversation.split('```')[num_csv - 1]
        csv_source = io.StringIO(dataset_csv)
        df = pd.read_table(csv_source, sep=",", index_col=1, skipinitialspace=True)
        state.content.dataframe = df
        st.dataframe(state.content.dataframe)
    
    if 'Vegalite Chart Generator' in state.chat.idea and '$schema' in state.chat.conversation:
        # capture the chart json from second code fenced block
        # and assign it to state.content.vegalite
        num_jsons = state.chat.conversation.count('```')
        chart_json = state.chat.conversation.split('```')[num_jsons - 1]
        state.content.vegalite = json.loads(chart_json.replace('vega-lite {', '{'))
        redundant_instruction = state.chat.conversation.find('You can copy and paste this code into a Vega-Lite editor')
        if redundant_instruction != -1:
            state.chat.conversation = state.chat.conversation[:redundant_instruction]
        st.vega_lite_chart(state.content.vegalite)

    if 'youtube.com' in state.chat.conversation:
        # create a list of youtube links
        youtube_links = [link for link in state.chat.conversation.split() if 'youtube.com' in link]
        youtube_url = youtube_links[-1]
        st.video(youtube_url)        
        state.content.youtube = youtube_url

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
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=state.chat.messages,
                max_tokens=500,
                temperature=0)

            state.open_ai.chatgpt_runs += 1
            state.open_ai.tokens += response.usage.total_tokens
            # append the response to the conversation
            state.chat.conversation += '\n' + 'Assistant: ' + response.choices[0].message.content
            # force render the page
            state.ux.code = True
            st.experimental_rerun()

if state.ux.code:
    st.sidebar.markdown("### 🪄 Get code")
    st.sidebar.button('Generate tutorial with code', on_click=generate_code)
