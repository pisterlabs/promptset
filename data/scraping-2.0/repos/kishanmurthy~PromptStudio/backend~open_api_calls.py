from langchain.llms import OpenAI

def generate_openai_output(node_prompt, node_output_format):
    print(node_prompt)
    llm = OpenAI(openai_api_key="")
    output = llm.predict(node_prompt)
    return output


# generate_openai_output("what are the colors in rainbow", "JSON")