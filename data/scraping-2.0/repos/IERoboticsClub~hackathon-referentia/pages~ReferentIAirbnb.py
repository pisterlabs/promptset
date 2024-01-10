import streamlit as st
from PIL import Image
import os
import openai
from utils.redis_helpers import connect_redis, reformat_redis, upload_reviews_to_redis, create_reviews_query_context
from utils.nlp_helpers import get_reviews_sentences, get_db_schema
from utils.scrapper import Airbnb_scraper
from streamlit_chat import message
from selenium import webdriver
import json
from dotenv import load_dotenv

load_dotenv()

# ------ PAGE CONFIG ------
st.set_page_config(
    page_title="Airbnb"
)

# ------ CONSTANTS ------
redis_conn = connect_redis()
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

# Make sure to have driver 'chromedriver.exe' in folder 'driver'
driver = webdriver.Chrome(
    executable_path=r'driver\chromedriver.exe',
    chrome_options=Airbnb_scraper.chrome_options())

# ------ SIDEBAR ------
with st.sidebar:

    st.write("# GPT Model Parameters")
    max_tokens = st.number_input("Max Tokens", value=500, step=100)
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    presence_penalty = st.slider("Presence Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    frequency_penalty = st.slider("Frequency Penalty", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)


# ------ MAIN PAGE ------
col1, col2, col3 = st.columns([4, 1, 1])
col1.header("Airbnb Review's Chatbot")
retrieved_url = st.text_input("Insert a link to an Airbnb listing", "https://www.airbnb.es/rooms/plus/23840102?adults=1&category_tag=Tag%3A7769&children=0&enable_m3_private_room=true&infants=0&pets=0&search_mode=flex_destinations_search&check_in=2023-10-16&check_out=2023-10-21&federated_search_id=42f775af-840a-450e-907c-f0fc91b059f6&source_impression_id=p3_1685434270_ulStyYBL9a3RL5Ax")
left, mid, right = st.columns([2, 2, 1])
if left.button("Restart Redis"):
    try:
        redis_conn = connect_redis()
        reformat_redis(redis_conn)
        st.info('Successfully formatted Redis', icon="ℹ️")
    except:
        st.error("RedisError: Please check the connection with the Redis server!")

if mid.button("Scrape"):
    scrapper = Airbnb_scraper(driver)
    try:
        reviews = scrapper.get_reviews(retrieved_url)
        if reviews != None:
            # Save reviews dictionary to file
            with open('./tempDir/reviews.json', 'w') as fp:
                json.dump(reviews, fp)
            st.info('Successfully scraped reviews', icon="ℹ️")
            with st.expander("Show scraped data"):
                st.write(reviews)
        else:
            st.warning("Error: Seems like the airbnb doesn't meet the requirements!")
    except:
        reviews = None
        st.warning("Error: Please check the link you inserted!")   

if right.button("Upload data"):
    # Load reviews dictionary from file
    try:
        with open('./tempDir/reviews.json', 'r') as fp:
            reviews = json.load(fp)
    except:
        st.warning("Error: Please scrape the data first!")
        reviews = None
    if reviews is not None:
        reviews = reviews['Reviews']
        sentences = get_reviews_sentences(reviews)
        db_schema = get_db_schema(sentences)
        # Schema: {id: [id, sentence, embedding]}
        #print("Successfully processed")
        #print("Uploading data to Redis")
        for key, value in db_schema.items():
            upload_reviews_to_redis(
                value[0], 
                value[1],
                value[2],
                redis_conn)
        print("Index size: ", redis_conn.ft().info()['num_docs'])
        st.info('Successfully uploaded reviews to Redis database', icon="ℹ️")
    else:
        st.write("No reviews uploaded!")


st.header("")
user_query = st.text_area('Ask questions about the apartment', '')
left, right = st.columns([3,1])

if right.button("Clear Chat History", key="clear"):
    st.session_state['generated'] = []
    st.session_state['past'] = []
if left.button("Submit"):
    try:
        assistant_prompt = create_reviews_query_context(redis_conn, user_query)
    except:
        st.warning("Error loading your data: Please check the connection with Redis")
        assistant_prompt = user_query
    with st.expander("See generated prompt"):
        st.text(assistant_prompt)           
    try:
        openai.api_version =  None
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=assistant_prompt,
            max_tokens=max_tokens
        )
        output = response.choices[0].text
    except:
        st.warning("Error with the OpenAI API!")
        output = "None"
    st.session_state.past.append(user_query)
    if output != "None":
        st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')    