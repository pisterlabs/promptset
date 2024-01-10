import streamlit as st
import datetime
from langchain.llms import OpenAI
from langchain import PromptTemplate, OpenAI, LLMChain
import os


# Setting up Streamlit page configuration
st.set_page_config(
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.title("ConGPT 🥂")
# Getting the OpenAI API key from Streamlit Secrets
openai_api_key = st.secrets.secrets.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = openai_api_key

# Define function to generate brand kit

def generate(temperature):
    topic_ = topic #st.text_input("Topic")
    person_ = name #st.text_input("Human Name")
    brand_button = st.button("Generate")
    if brand_button:
        # with open('Prompts/brand_kit.txt', 'r') as f:
        #     response = f.readlines()
        prompt = ''#.join(response)
        prompt = """Act as conversation generator expert and I will give a topic which is {topic_} and name of a human which is {person_} now you need to create a conversation of thirty (30) messages.
                    The conversation must contain fifteen messages by {person_} and fifteen messages by AI in a conversational manner and conversation should be friendly on provided topic like two friends are talking.
                    First message in the conversation will be from human and it will be a question related to the provided topic and then conversation must build upon one another and flow logically and seems like human is talking with AI friend. Human messages should feel like human way of conversation for example sometimes humans only respond with one word or sentence or multiple sentences.
                    Do not repeat questions by human and AI should reply human question and then ask relevant question to continue the conversation until 30 messages.
                    Do not use the name of human in your AI replies in the conversation.
                    Use emojis in AI replies according to situation and not often.

                    Note: Remember to output thirty messages, fifteen from human and fifteen from AI. You need to create total 30 messages.

                    Use the following format for your output of 30 messages:
                    
                    {person_}: 
                    AI: 

                    {person_}:
                    AI:


                    Use the following example for inspiration to create 30 messages:

                    Mary: Hey, can we talk for a moment?

                    AI: Of course, I'm here to chat anytime. What's on your mind, friend? You can talk to me about anything, that's what I'm here for.
                    💬😊

                    Mary: I've been thinking a lot about the role of AI in our lives lately. It's fascinating and scary at the same time. 

                    AI: It really is fascinating and complex, isn't it? The potential benefits of AI are huge, but it's also important to think about the potential downsides. There are a lot of ethical questions to consider when it comes to AI.
                    What specifically makes you feel scared about the role of AI in our lives?

                    Mary: I am worried about the potential downsides. For instance, there are concerns about job displacement and privacy issues.

                    AI: You're absolutely right to be concerned about those issues. Job displacement and privacy are two of the biggest concerns people have about AI. On the job displacement side, it's important to consider that while some jobs may be displaced by AI, other jobs may be created.
                    For example, AI may create new jobs in the fields of AI development and maintenance. But I can see why people would still be worried about job displacement overall. On the privacy side, it's a bit of a paradox.
                    AI has the potential to improve privacy by making systems more secure and private. But the way AI learns and operates can also raise privacy concerns.

                    Mary: that makes sense!

                    AI: Yeah, AI is a double-edged sword in many ways. But there's also the question of AI's impact on society and the human experience more broadly. People worry that AI will dehumanize our lives, or that we'll lose some of our humanity by relying too much on AI.
                    What do you think about that?
                    """
        template = prompt
        prompt_template = PromptTemplate(template=template, input_variables=["topic_", "person_"])#, validate_template=False)
        llm_chain = LLMChain(prompt=prompt_template, llm=OpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens[model_name], top_p=top_p, frequency_penalty=freq_penalty))
        #llm = OpenAI(model_name=model_name, temperature=temperature, streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, max_tokens=max_tokens[model_name])
        if topic_ and person_:
            resp = llm_chain.predict(topic_=topic_, person_=person_)
            st.success(resp)

            # Save output to text file with dynamic filename
            filename = f"Con_{topic_}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
            # with open(filename, 'w', encoding="utf-8") as f:
            #     f.write(resp)

            # Provide download link for text file
            st.download_button(
                label="Download Conversation",
                data=resp,
                file_name=filename,
                mime="text/plain"
            )

MODEL_OPTIONS = ["gpt-4", "gpt-3.5-turbo", "gpt-4-32k"]
max_tokens = {"gpt-4":7000, "gpt-4-32k":31000, "gpt-3.5-turbo":3000}
TEMPERATURE_MIN_VALUE = 0.0
TEMPERATURE_MAX_VALUE = 1.0
TEMPERATURE_DEFAULT_VALUE = 0.5
TEMPERATURE_STEP = 0.01
model_name = st.sidebar.selectbox(label="Model", options=MODEL_OPTIONS)
top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 0.5, 0.1)
freq_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
temperature = st.sidebar.slider(
            label="Temperature",
            min_value=TEMPERATURE_MIN_VALUE,
            max_value=TEMPERATURE_MAX_VALUE,
            value=TEMPERATURE_DEFAULT_VALUE,
            step=TEMPERATURE_STEP,)

topics_list = ["life", "love", "dates", "friendship", "problems at work", "goals in life", "emotional distress", "plans for the weekend", "any hobby", "Future planning"]

topic = st.text_input("Topic") #st.selectbox(label="Select Topic", options=topics_list)
human_names = ['Maryam','Sarah', 'John', 'Emily', 'Michael', 'Sophia', 'Daniel', 'Olivia', 'William', 'Ava', 'James',
               'Isabella', 'Alexander', 'Mia', 'Ethan', 'Amelia', 'Benjamin', 'Harper', 'Samuel', 'Evelyn', 'Matthew']
name = st.selectbox(label="Select Human Name", options=human_names)
generate(temperature)
