import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, filename="test3.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')

llm = OpenAI(openai_api_key=OPEN_AI_KEY)

prompt_templates = {
    "product": PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    ),
    "service": PromptTemplate(
        input_variables=["service"],
        template="What is a good name for a company that provides {service}?",
    )
}

while True:
    # Ask user for template selection
    template_name = input("Please select a template (product, service): ")

    # Check if chosen template exists
    if template_name not in prompt_templates:
        print("Invalid template. Please choose 'product' or 'service'.")
        continue

    chosen_template = prompt_templates[template_name]

    # Ask user for input for the chosen template
    user_input = input(f"Please provide a {template_name} description: ")

    chain = LLMChain(llm=llm, prompt=chosen_template)

    response = chain.run(**{template_name: user_input})

    # Log and print the response
    logging.info(f"Generated response: {response}")
    print(f"Generated response: {response}")
