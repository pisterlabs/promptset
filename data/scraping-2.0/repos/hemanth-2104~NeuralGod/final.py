import streamlit as st
import os
import subprocess
from openai import OpenAI
import openai
os.environ['OPENAI_API_KEY'] ='<your_openai_key>'
openai.api_key = '<your_openai_key>'

client = OpenAI()
a = None
global b
b = None
def check_vul(text, typey):
    # print(text, typey)
    if typey == 'None':
        ini = "FQDC"
    elif typey == 'FFMPEG' or typey == "QEMU":
        ini = "FQ"
    elif typey == 'Debian' or typey == 'Chrome':
        ini = "DC"
    else:
        ini = "AL"
    # print(ini)
    with open(f'./{ini}/input.cpp', 'w') as file:
        file.write(str(text))
    original_directory = os.getcwd()
    os.chdir(ini)

    # Run the command "python main.py -code input.cpp"
    command = "python main.py -code input.cpp"
    subprocess.run(command, shell=True)
    os.chdir(original_directory)
    with open(f'./{ini}/output.txt', 'r') as file:
        num = file.readline()[-2]
    return num

def fix_vul(ins):
    prompt = f'''Please provide an improved version of the input code that addresses potential vulnerabilities and follows best coding practices and also add comment for the new change.
                
                {ins}
        
                Give me only the corrected code, without any more text
                '''
    print(prompt)


    response = client.completions.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=500,
    temperature=0.6,
    n=1,
    stop=None,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    # print('reached')
    corrected_code = response.choices[0].text.strip()
    print()
    print(corrected_code)
    return corrected_code


st.set_page_config(
    page_title="Neural God",
    page_icon=":brain:",
    layout="wide",
)

col1, col2 = st.columns([2, 3])


st.header("Neural God")
st.subheader("Code Vulnerability Detection")

uploaded_file = st.file_uploader("Upload a C/C++ file (.c or .cpp)", type=["c", "cpp"])
if st.button("Submit Code"):
    if uploaded_file:
        code_input = uploaded_file.read().decode("utf-8")
        # print(code_input)
        if len(code_input) == 0:
            st.warning("Input the proper code")
        else:
            st.success("File submitted successfully.")

option = st.selectbox("Select an Option", ["None", "FFMPEG", "QEMU", "Debian", "Chrome","Android", "Linux"])
if st.button("Choose Code Type"):
    st.success("Code type selected successfully")

if st.button("Check Vulnerability"):
    st.text("Checking for vulnerabilities...")
    a = check_vul(uploaded_file.read().decode("utf-8"), option)
    # a = 1
    a = int(a)
    st.success("Vulenerability checked")
    st.text(f"Your Code is {'Vulenrable' if a == 1 else 'Not Vulenrable'}")
if st.button("Fix Code"):
    st.text("fixing code")
    b = fix_vul(uploaded_file.read().decode("utf-8"))
    st.code(b)