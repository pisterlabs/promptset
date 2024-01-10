import os
import numpy as np
import streamlit as st
import openai
from dotenv import load_dotenv
from streamlit_chat import message
from PIL import Image

# Constants
SENTENCES_PER_EPISODE = 5   # number of sentences per episode
RIDDLE_MIN = 2 # minimum number for riddles
MODEL = "gpt-3.5-turbo-0613" # use a better performing gpt model if possible

# Load OpenAI key from .env file
load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_KEY")

# Load image
image = Image.open('children.png')

def reset_state():
    """Resets the session state."""
    keys = list(st.session_state.keys())
    for key in keys:
        del st.session_state[key]

def generate_riddle(calculation_type, riddle_max):
    """Generates a riddle and returns the question and answer."""
    num1 = np.random.randint(RIDDLE_MIN, riddle_max)
    num2 = np.random.randint(RIDDLE_MIN, riddle_max)
    if calculation_type == "Addition":
        question = "{} + {}".format(num1, num2)
        answer = num1 + num2
    elif calculation_type == "Subtraction":
        question = "{} - {}".format(max(num1, num2), min(num1, num2))
        answer = max(num1, num2) - min(num1, num2)
    elif calculation_type == "Multiplication":
        question = "{} * {}".format(num1, num2)
        answer = num1 * num2
    elif calculation_type == "Division":
        product = num1 * num2
        question = "{} / {}".format(product, num1)
        answer = num2
    return question, answer

def generate_story(messages):
    """Generates a story episode using the OpenAI API and returns the story."""
    story = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.5,
    )
    return story['choices'][0]['message']['content']

def generate_challenge():
    """Generates a set of riddles and stories for the child to solve."""
    st.session_state['right_answer'] = [] # list of right answers
    st.session_state['question'] = [] # list of questions
    st.session_state['story'] = [] # list of episodes  
    st.session_state['num_riddles'] = st.session_state['riddle_count'] # number of riddles persistent in session state
    messages = [] # list of messages for openai requests

    # system message for openai requests
    sys_message = """Tell a seven-year-old child a continuation story about {2}. Each episode consists of exactly {0} sentences. The story is about the day of {1}. An episode of the story consists of exactly {0} sentences and no more. Start directly with the narration. End each episode with a math problem, which is always posed by [role: user] beforehand. Integrate the math problem into the narration of the episode. Make sure the math problem is correctly formulated. Do not give the solution. By solving this problem, the child can help {1}. Continue in the new episode already told episodes and pose a new math problem. PLEASE NOTE: Do not give the solution to the math problem. Use only {0} sentences. End the end with the math problem.""".format(SENTENCES_PER_EPISODE, st.session_state['person'], st.session_state['topic'])
    
    messages.append({"role": "system", "content": sys_message})

    # Create a progress bar
    progress_bar = st.progress(0)
    status_message = st.empty()
    status_message.text("I am generating your story...")
    for i in range(st.session_state['riddle_count']): # generate riddles and stories
        # Update the progress bar
        progress_bar.progress((i + 1) / st.session_state['riddle_count'])

        # generate riddle
        calculation_type = np.random.choice(st.session_state['calculation_type'])
        question, answer = generate_riddle(calculation_type, st.session_state['riddle_max'])
        messages.append({"role": "user", "content": question})

        # generate story
        story = generate_story(messages)
        messages.append({"role": "assistant", "content": story})
        
        # save riddle and story to session state
        st.session_state.right_answer.append(answer)
        st.session_state.question.append(question)
        st.session_state.story.append(story)

    # create final episode
    messages.pop(0) # remove first item in messages list (system message)
    messages.append({"role": "user", "content": "Finish the story in five sentences. Do not include a math problem."})
    story = generate_story(messages)
    st.session_state.story.append(story)   
    
    st.session_state['current_task'] = 0 # keeps track of the current episode
    status_message.empty()  # remove the status message
    return st.session_state['story'][0] # return first episode

def on_input_change():
    """Handles child input and checks if it is correct."""
    user_input = st.session_state["user_input"+str(st.session_state['current_task'])] # get user input
    st.session_state['past'].append(user_input) # save user input to session state
    if user_input == st.session_state.right_answer[st.session_state['current_task']]: # user input is correct
        #check if all tasks done
        if st.session_state['current_task'] == st.session_state['num_riddles']-1: # all tasks are done
            st.session_state['generated'].append(st.session_state['story'][st.session_state['current_task']+1]) # generate final episode
            st.session_state['finished'] = True # set finished flag
        else: # not all tasks are done
            st.session_state['current_task']+=1 # increase current task counter
            st.session_state['generated'].append(st.session_state['story'][st.session_state['current_task']]) # append next episode for output
    else: # user input is wrong
        st.session_state.generated.append("Not quite right. Try again! " + st.session_state.question[st.session_state['current_task'] ])  # append wrong message to output

if 'end_story' not in st.session_state:
    st.session_state['end_story'] = False

if 'input_done' not in st.session_state:
    st.session_state['input_done'] = False

st.title("༼ ͡ಠ ͜ʖ ͡ಠ ༽ Your Math Adventure")
st.image(image, use_column_width=True, caption = "3 * 2 = 7 ???")

if st.session_state['end_story']: # story ended
    st.write("The story has ended.")
    if st.button("Start a new story"):
        reset_state()
else: # story not ended
    if st.session_state['input_done'] == False:
        with st.sidebar:
            st.selectbox("How many math problems would you like to solve?", [3, 5, 7, 10], key="riddle_count", index=0)
            st.multiselect("Choose the calculation type", ["Addition", "Subtraction", "Multiplication", "Division"], key="calculation_type", default=["Addition", "Subtraction", "Multiplication", "Division"])
            st.selectbox("Choose the number range", ["1 digit (1-9)", "2 digits (1-99)"], key="number_range", index=0)
            st.text_input("Provide a character for your story", key="person")
            st.text_input("Provide a topic for your story", key="topic")
            if st.button("Start the story", key="start_btn"):
                st.session_state['input_done'] = True
                if st.session_state['number_range'] == "1 digit (1-9)":
                    st.session_state['riddle_max'] = 9
                else:
                    st.session_state['riddle_max'] = 99

    if st.session_state['input_done']:
        if 'past' not in st.session_state:
            st.session_state['past']=['Here your answers are shown.']
        if 'generated' not in st.session_state:
            st.session_state['generated'] = [generate_challenge()]
        if 'finished' not in st.session_state:
            st.session_state['finished'] = False
        chat_placeholder = st.empty()
        with chat_placeholder.container():    
            # st.write(st.session_state.story) # for debugging
            for i in range(len(st.session_state['generated'])):  
                            
                message(str(st.session_state['past'][i]), is_user=True, key=str(i) + '_user')
                message(
                    st.session_state['generated'][i],
                    key=str(i)
                )

        if not st.session_state['finished']:
            with st.container():
                st.number_input("Your solution:", min_value=-1, max_value=100, 
                                value=-1, step=1, on_change=on_input_change, 
                                key="user_input"+str(st.session_state['current_task']))
        if st.button("End the story", key="end_btn"):
            st.session_state['end_story'] = True
