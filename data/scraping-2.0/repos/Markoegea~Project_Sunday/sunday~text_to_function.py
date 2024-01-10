import json
from langchain.llms.ctransformers import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def get_function_from_text(text:str):

    print(text)

    template = """
    USER: <<question>> {text} <<function>> {functions_string}\nASSISTANT: 
    """
    with open('sunday/functions.json', 'r') as file:
        functions = json.load(file)

    prompt = PromptTemplate(input_variables=['text', 'functions_string'], template=template)

    llm = CTransformers(model="TheBloke/gorilla-openfunctions-v1-GGUF",
                        model_file="gorilla-openfunctions-v1.Q4_K_M.gguf",
                        model_type="llama", 
                        config={'context_length':-1}, # Change the value to your choice
                        gpu_layers=20, # Use 0 if you don't want to use GPU
                        temperature=0.2) # Use the minimum value to avoid the model imagine things

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    response = llm_chain.run(text=text, functions_string=json.dumps(functions))

    return response

if __name__ == '__main__':
    print(get_function_from_text('Print the value Hello World!'))