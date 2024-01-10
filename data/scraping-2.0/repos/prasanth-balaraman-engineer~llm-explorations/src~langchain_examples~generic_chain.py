import re

from dotenv import load_dotenv
from langchain.chains import TransformChain, LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()


def transform_func(inputs: dict) -> dict:
    text = inputs["text"]

    # replace multiple new lines and multiple spaces with a single one
    text = re.sub(r"(\r\n|\r|\n){2,}", r"\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return dict(output_text=text)


template = """Paraphrase this text:

{output_text}

In the style of a {style}.

Paraphrase: 
"""

input_text = """
Chains allow us to combine multiple 


components together to create a single, coherent application. 

For example, we can create a chain that takes user input,       format it with a PromptTemplate, 

and then passes the formatted response to an LLM. We can build more complex chains by combining     multiple chains together, or by 


combining chains with other components.
"""

if __name__ == "__main__":
    clean_extra_spaces_chain = TransformChain(
        input_variables=["text"],
        output_variables=["output_text"],
        transform=transform_func,
    )
    llm = OpenAI(model_name="text-davinci-003")
    prompt = PromptTemplate(input_variables=["output_text", "style"], template=template)
    style_paraphrase_chain = LLMChain(llm=llm, prompt=prompt, output_key="final_output")
    sequential_chain = SequentialChain(
        chains=[clean_extra_spaces_chain, style_paraphrase_chain],
        input_variables=["text", "style"],
        output_variables=["final_output"],
    )
    result = sequential_chain.run(dict(text=input_text, style="a 90s rapper"))
    print(result)
