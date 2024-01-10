# Bring in deps
import os  
import ast
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

llm = OpenAI(temperature=0.5, max_tokens=1000) 
st.title('🦜🔗 Code Generator')

def generate_file_code(
    filename, language, files, prompt, 
):
    prompt_template=PromptTemplate(input_variables=['prompt', 'language', 'files', 'filename'], 
            template="""You are an AI developer working on a code generation program. Here are the details:
            App: {prompt}
            Files to generate: {files}
            Language: {language}

            Generate valid code for the specified file and type. Return only the code, without any explanations. Follow these guidelines:

            Generate code specifically for the file {filename}.
            Remember that the purpose of our app {prompt}, so every line of generated code must be valid.
            Start generating the code now.
            Important Instructions 
                - return only valid code for the given filepath and file type, and 
                - return only the code without any instructions or fencing
                - do not add any other explanation. 
                - Return the whole code without missing any part of the code
    """)
    response_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, output_key='code')
    # call openai api with this prompt
    filecode = response_chain.run(prompt=prompt, language= language, files= files, filename=filename)

    return filename, filecode

def generate_filenames(language, prompt):
    prompt_template = PromptTemplate(input_variables=['prompt', 'language'], template="""You are an AI developer working on a code generation program. Here are the details:
            App: {prompt}
            Language: {language}
            User Intent: {prompt}
            Your task is to create a comprehensive list of filepaths that the user would write for the program {prompt}.
            Output: Just return the list of filepaths as a Python list of strings without any explanation or extra text.""")
    
    response_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, output_key='files')
    # call openai api with this prompt
    filenames = response_chain.run(prompt=prompt, language=language)
    return filenames

def write_file(filename, filecode, directory):
    # Output the filename in blue color
    print("\033[94m" + filename + "\033[0m")
    print(filecode)

    file_path = directory + "/" + filename
    dir = os.path.dirname(file_path)
    os.makedirs(dir, exist_ok=True)

    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write content to the file
        file.write(filecode)


if __name__ == "__main__":
    directory = st.text_input("Enter the directory you want to create the program in:")
    language = st.text_input("Enter the language you want to use to develop the app")
    prompt = st.text_area("Enter your prompt to generate the app code:")

    if st.button("Generate App"):
        filenames = generate_filenames(prompt=prompt, language=language)
        filenames = ast.literal_eval(filenames)

        for file in filenames:
            filename, filecode = generate_file_code(filename=file,language=language, files=filenames, prompt=prompt)
            write_file(filename=filename, filecode=filecode, directory=directory)

        print("Code generated successfully")


