import streamlit as st
import openai
from streamlit_ace import st_ace
import subprocess
import pyautogui
import voice

def generate_script(input, prompt, robotc_path, script_path):
    """Generates the script for the robot to execute by calling the OpenAI API
    It also compiles the script and runs it on the robot

    Args:
        content (string): The code that is already written with robot description and overall structure. Boilerplate code.
        instructions_incode (string): string of instructions for robot in code. This will get appended to the content.
        instructions_prompt (string): string of codex edit instructions
    """
    response = openai.Edit.create(
        model="code-davinci-edit-001",
        input = f"{input}\n",
        instruction=prompt,
        temperature=0.5,
        top_p=1
    )

    if 'choices' in response:
        response_choices = response['choices']
        if len(response_choices) > 0:
            # Display the first choice
            st.code(response_choices[0]['text'], language="c")

            # save the script to a file
            with open('script.c', 'w') as f:
                f.write(response_choices[0]['text'])

            # Download the first choice
            st.download_button('Download Script', response_choices[0]['text'].encode('utf-8'), file_name='script.c', mime='text/plain')
            
            # Compile the script
            with st.spinner('Compiling...'):
                # Open RoboC and Compile the script
                subprocess.Popen(robotc_path)
                pyautogui.sleep(1)
                pyautogui.hotkey('ctrl', 'o') # Open file
                pyautogui.sleep(1)
                pyautogui.typewrite(script_path) # Type the path to the script
                pyautogui.sleep(2)
                pyautogui.press('enter') # Press enter
                pyautogui.sleep(3)
                pyautogui.press('f5') # Compile
                # pyautogui.sleep(11)
                # x, y = pyautogui.locateCenterOnScreen('robotc_start.png', confidence=0.9)
                # pyautogui.moveTo(x, y)
                # pyautogui.click()
                # pyautogui.sleep(5)
                # pyautogui.hotkey('alt', 'f5') # Close RobotC
            st.success('Done!')

        else:
            st.write("No choices found")

# Environment variables
robotc_path = r'C:\Program Files (x86)\Robomatter Inc\ROBOTC Development Environment 4.X\ROBOTC.exe' 
script_path = r'C:\coding\GitHub\Robo-autoscript\Dashboard\script.c'

st.set_page_config(
     page_title="Robo Auto Script",
     page_icon="🤖",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "Scripting RoboC code with OpenAi's Codex"
     }
)

openai.api_key = st.secrets["OPENAI_API_KEY"]

# select the boilerplate code
boilerplate = st.selectbox("Select the boilerplate code", ["2_wheel_drive", "4_wheel_drive"])

# Get the boilerplate code
with open(f'boilerplates/{boilerplate}.txt', 'r') as f:
    boilerplate = f.read()

# Display the boilerplate code for prompt engineering
boilerplate = st_ace(value=boilerplate, language="c_cpp")

# Tabs for mode selection
tab1, tab2, tab3 = st.tabs(["Type SI", "Type PS", "Voice SI"])

# Type Sequence of Instructions (SI) mode
with tab1:
    st.header("Type Sequence of Instructions (SI)")

    # Instruction input
    if 'instructions' not in st.session_state:
        st.session_state['instructions'] = ['Stop']

    new_instruction = st.text_input("Add Instruction")
    col1, col2 = st.columns(2)
    if col1.button("Add instruction"):
        st.session_state['instructions'].insert(-1, new_instruction)
    if col2.button("Clear Instructions"):
        st.session_state['instructions'] = ['Stop']
        st.experimental_rerun()

    # Prepare instructions for codex
    instructions_incode = ""
    instructions_prompt = "Edit and complete the code below to execute the instructions:\n"
    for index, instruction in enumerate(st.session_state['instructions']):
        st.caption(f"{index + 1}. {instruction}")
        instructions_prompt += f"    {index + 1}. {instruction}\n"
        instructions_incode += f"   // {index + 1}. {instruction}\n\n\n"
    instructions_input = boilerplate + instructions_incode

    # Generate code
    if st.button("🤖 Generate Script", key="TSI_script"):
        generate_script(instructions_input, instructions_prompt, robotc_path, script_path)

with tab2:
    st.header("Type Problem Solving (PS)")

    # Problem input
    problem_prompt = st.text_area("Problem")

    # Generate code
    if st.button("🤖 Generate Script", key='TPS_script'):
        generate_script(boilerplate, problem_prompt, robotc_path, script_path)


# Voice Sequence of Instructions mode          
with tab3:
    st.header("Voice Sequence of Instructions (SI)")

    recording = st.button("🎤 Start Recording")
    if recording:
        instructions = voice.voice_to_instructions()

        # Prepare instructions for codex
        instructions_incode = ""
        instructions_prompt = "Edit and complete the code below to execute the instructions:\n"
        for index, instruction in enumerate(instructions):
            st.caption(f"{index + 1}. {instruction}")
            instructions_prompt += f"    {index + 1}. {instruction}\n"
            instructions_incode += f"   // {index + 1}. {instruction}\n\n\n"
        instructions_input = boilerplate + instructions_incode

        # Generate code
        generate_script(instructions_input, instructions_prompt, robotc_path, script_path)