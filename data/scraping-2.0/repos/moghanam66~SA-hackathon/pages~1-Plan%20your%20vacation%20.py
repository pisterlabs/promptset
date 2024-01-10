import streamlit as st
import pandas as pd
import re
from deep_translator import GoogleTranslator
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain,SequentialChain
from langchain.prompts import ChatPromptTemplate
from utils import *

def main():

    # Congfiger the page attributes
    st.set_page_config(
        page_title='Plan your vacation',
        page_icon=":star:",layout="wide")

    # Set the background and the logo
    set_background('wallpaper.jpeg')
    add_logo("logo4.png")
    # Set the title
    st.title('Tell us what you want to do and let AI plan your stay for you.')

    # Read the scrapped data of the attraction sites
    df=pd.read_csv('attractionsEng.csv')

    # Get all the names of the attraction sites
    placesDf=list(df['attractionSite'])

    # Ask the user what he wants to do 
    userInput = st.text_input(" What do you want to do in Saudi ?",'I want to visit fun sites')

    if userInput !='':
        #Get the plane and the places to visit that chat GBT introduced
        answer=openAiPlaner(userInput,placesDf,OPENAI_API_KEY)
        plane=answer['plane']
        places=answer['places'].split('\n')

        # Split the plans into days and translate them into Arabic if the question is in Arabic
        days=plane.split('\n\n')
        for day in days:
            if re.fullmatch('^[\u0621-\u064A0-9 ]+$',userInput):
                dayAr=GoogleTranslator(source='en',target='ar').translate(day).replace('-','\n-')
                st.markdown("<div style='direction: RTL;'> {} </div>".format(dayAr), unsafe_allow_html=True)
            else:
                st.markdown(day)

            # Get all the places mentioned in each day to extract images of the same places
            placesToDisplay=[place[3:].strip() for place in places if place[3:].strip() in day]
            images=[]
            captions=[]
            for placeToDisplay in placesToDisplay:
                image=df[df['attractionSite']==placeToDisplay]['image'].values

                if len(image)>0:
                    # Translate the caption into Arabic if the question is in Arabic
                    if re.fullmatch('^[\u0621-\u064A0-9 ]+$',userInput):
                        placeToDisplay=GoogleTranslator(source='en',target='ar').translate(placeToDisplay)
                    images.append(image[0])
                    captions.append(placeToDisplay)

            # Display the images of the places mentioned on this day
            st.image(images,caption=captions,width=300)

# Cache the answer to optimize the performance and reduce any wasted cost. 
@st.cache_data
def openAiPlaner(question,placesDf,API_KEY):

    #Intialize the LLM
    llm = ChatOpenAI(openai_api_key=API_KEY,temperature=0)

    # prompt template 1: Get all the suitable places
    first_prompt = ChatPromptTemplate.from_template(
        "This is a question from a tourist visiting Saudi Arabia:"
        "\n\n{Question}"
        f"\n\n Suggest 10 places to visit from this list{placesDf}")

    # chain 1: input= Question and output= places
    chain_one = LLMChain(llm=llm, prompt=first_prompt,
                        output_key="places")
    
    # prompt template 2: Create a plan to visit those places
    second_prompt = ChatPromptTemplate.from_template(
        "Create a plan to visit those places:"
        "\n\n{places}")
    
    # chain 2: input= places and output= plan
    chain_two = LLMChain(llm=llm, prompt=second_prompt,
                        output_key="plane")
    
    #Include all chains and create the sequential chain
    overall_chain = SequentialChain(chains=[chain_one,chain_two],
    input_variables=["Question"],
    output_variables=["places",'plane'],
    verbose=False)

    return overall_chain(question)

if __name__ == "__main__":
    main()    
