import streamlit as st
import streamlit.components.v1 as components
from textblob import TextBlob
from PIL import Image
import text2emotion as te
import plotly.graph_objects as go
import json
import re

# langchain 
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.llms import OpenAI

# load model 
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


def plotPie(labels, values):
    fig = go.Figure(
        go.Pie(
        labels = labels,
        values = values,
        hoverinfo = "label+percent",
        textinfo = "value"
    ))
    st.plotly_chart(fig)

    
def getPolarity(userText):
    tb = TextBlob(userText)
    polarity = round(tb.polarity, 2)
    subjectivity = round(tb.subjectivity, 2)
    if polarity>0:
        return polarity, subjectivity, "Positive"
    elif polarity==0:
        return polarity, subjectivity, "Neutral"
    else:
        return polarity, subjectivity, "Negative"


def gpttoTextBlod(userText, polarity, subjectivity):
    # define the output
    response_schemas = [
        ResponseSchema(name="sentiment", description="a sentiment label based on the user text. It should be either Negative, Positive or Neutral"),
        ResponseSchema(name="reason", description="""
        If the sentiment is Negative then return the reason why the user shouldn't have said that.
        If the sentiment is Positive then return a compliment.
        For Neutral then return a instruct for a better message. 
        """),
        ResponseSchema(name="Botideas", description="the best and friendliest replacement to the given user text")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # prompt template
    template = """You are good at detecting human emotion. All emotions you know are Negative, Positive and Neutral.
    Given a human text, subjectivity and polarity, your job is to answer as best as possible.
    Know that subjectivity is a measure of how much of a text is based on personal opinions or beliefs, rather than on facts. 
    It is a float value between 0 and 1, where 0 represents an objective text and 1 represents a highly subjective text.
    Also know that polarity is a indicator for the sentiment of the given user text, negative value means Negative, positive value means Positive and 0 means Neutral.
    {format_instructions}
    User text: {text}
    Subjectivity: {subjectivity}
    Polarity: {polarity}"""
    
    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(template=template, input_variables=["text","subjectivity","polarity"],
                            partial_variables={"format_instructions": format_instructions})
    model = OpenAI(verbose=True, temperature=0.0)
    # Build chain
    sentiment_chain = LLMChain(llm=model, prompt=prompt, output_key='result')

    format_instructions = output_parser.get_format_instructions()
    prompt = PromptTemplate(template=template, input_variables=["text","subjectivity","polarity"],
                            partial_variables={"format_instructions": format_instructions})
    model = OpenAI(verbose=True, temperature=0.0)
    # Build chain
    sentiment_chain = LLMChain(llm=model, prompt=prompt, output_key='result')


    ans = sentiment_chain({"text": userText, "polarity": polarity, "subjectivity": subjectivity})
    return ans    
    
    
    

# process data json
def parse_nested_json(text):
    a = text.strip()
    json_data = a.strip().replace('```json', '').strip()
    json_data = json_data.strip().replace('```', '').strip()
    data = json.loads(json_data)
    return data    

# text to emotion gpt
def emotiongpt(user_input):
    response_schemas = []
    emos = ['Happy 😊','Sad 😔','Angry 😠','Surprise 😲','Fear 😨']

    for emo in emos:
        emos = emo.split(" ")
        
        des = f"""a js object contains 3 properties:
        "label": str // always return '{emos[1]}' 
        "score": int // an emotional score from 1 to 100 for the {emos[0]}ness based the given user text
        "reason": str// a reason for the score"""
        #print(des)
        schema = ResponseSchema(name=emos[0], description=des)
        response_schemas.append(schema)
        
    output_icon_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # prompt template
    template = """You are good at detecting human emotion, so good that you can measure the percentages of each emotion in Happy, Sad, Angry, Surprise, Fear from a given text. 
    The sum of the percentages of each emotion in Happy, Sad, Angry, Surprise, Fear must be 100.
    Given an user text, your job is to answer as best as possible.
    {format_instructions}
    User text: {text}. This is the end of the user text."""

    format_instructions = output_icon_parser.get_format_instructions()
    prompt = PromptTemplate(template=template, input_variables=["text"],
                            partial_variables={"format_instructions": format_instructions})

    # Build chain
    model = OpenAI(verbose=True, temperature=0.0)
    sentiment_icon_chain = LLMChain(llm=model, prompt=prompt, output_key='result')

    ans = sentiment_icon_chain({"text": user_input})
    
    return parse_nested_json(ans['result'])

    


def getSentiments(userText, type):
    if(type == 'Positive/Negative/Neutral-TextBlob'):
        polarity, subjectivity, status = getPolarity(userText)
       
        folder = os.path.dirname(__file__)+"/images"
        if(status=='Positive'):
            image = Image.open(f'{folder}/positive.PNG')
        elif(status == "Negative"):
            image = Image.open(f'{folder}/negative.PNG')
        else:
            image = Image.open(f'{folder}/neutral.PNG')
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Polarity", polarity, None)
        col2.metric("Subjectivity", subjectivity, None)
        col3.metric("Result", status, None)
        
        # next
        gptcomment = gpttoTextBlod(userText,  polarity=polarity, subjectivity=subjectivity)
        imgcol , commentcol = st.columns([1, 3])
        commentcol.write(gptcomment)
        imgcol.image(image, caption=status)
        
            
    elif(type == 'text2emotion'):
        
        data = emotiongpt(userText)
        st.write(data)
        emotion = dict(te.get_emotion(userText))
        print(emotion)
        
        # # print(emotion)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Happy 😊", data['Happy']['score'], None)
        col2.metric("Sad 😔", data['Sad']['score'], None)
        col3.metric("Angry 😠", data['Angry']['score'], None)
        col4.metric("Fear 😨", data['Fear']['score'], None)
        col5.metric("Surprise 😲", data['Surprise']['score'], None)
        
        
        emos_list = ['Happy 😊','Sad 😔','Angry 😠','Surprise 😲','Fear 😨']
        emos_values = [data['Happy']['score'], data['Sad']['score'],  data['Angry']['score'],data['Fear']['score'],data['Surprise']['score']]
        plotPie(emos_list,emos_values)
        
def renderPage():
    st.title("Heal.AI")
    folder = os.path.dirname(__file__)+"/images"
    image = Image.open(f'{folder}/textSEN.png')
    st.image(image)

    components.html("""<hr style="height:3px;border:none;color:#333;background-color:#333; margin-bottom: 10px" /> """)
    # st.markdown("### User Input Text Analysis")
    st.subheader("User Input Text Analysis")
    st.text("Analyzing text data given by the user and find sentiments.")
    st.text("")
    userText = st.text_input('User Input', placeholder='Input text HERE')
    st.text("")
    type = st.selectbox(
     'Type of analysis',
     ('Positive/Negative/Neutral-TextBlob', 'text2emotion'))
    st.text("")
    if st.button('Predict'):
        if(userText!="" and type!=None):
            st.text("")
            st.components.v1.html("""
                                <h3 style="color: #0284c7; font-family: Source Sans Pro, sans-serif; font-size: 28px; margin-bottom: 10px; margin-top: 50px;">Result</h3>
                                """, height=100)
            getSentiments(userText, type)
            
            
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 