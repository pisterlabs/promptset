# This method is effective for managing a large number of examples. It offers customization through various selectors, but it involves manual creation and selection of examples, which might not be ideal for every application type.

# Example of employing LangChain's SemanticSimilarityExampleSelector for selecting examples based on their semantic resemblance to the input. This illustration showcases the process of creating an ExampleSelector, generating a prompt using a few-shot approach:


from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import DeepLake
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

activeloop_token = os.getenv("ACTIVELOOP_TOKEN")

# Create a PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# Define some examples

examples = [
    {"input": "0°C", "output": "32°F"},
    {"input": "10°C", "output": "50°F"},
    {"input": "20°C", "output": "68°F"},
    {"input": "30°C", "output": "86°F"},
    {"input": "40°C", "output": "104°F"},
]

# create Deep Lake dataset
# TODO: use your organization id here.  (by default, org id is your username)

my_activeloop_org_id = "abduljaweed" 
my_activeloop_dataset_name = "langchain_course_fewshot_selector"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path)

# Embedding function

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Instantiate SemanticSimilarityExampleSelector using the examples

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=examples,
    embeddings=embeddings,
    db,
    k=1
)


# Create a FewShotPromptTemplate using the example_selector

similar_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Convert the temperature from Celsius to Fahrenheit",
    suffix="Input: {temperature}\nOutput:", 
    input_variables=["temperature"],
)

# Test the similar_prompt with different inputs

print(similar_prompt.format(temperature="10°C"))   # Test with an input
print(similar_prompt.format(temperature="30°C"))  # Test with another input

# Add a new example to the SemanticSimilarityExampleSelector

similar_prompt.example_selector.add_example({"input": "50°C", "output": "122°F"})
print(similar_prompt.format(temperature="40°C")) # Test with a new input after adding the example


# Keep in mind that the SemanticSimilarityExampleSelectoruses the Deep Lake vector store and OpenAIEmbeddingsto measure semantic similarity. It stores the samples on the database in the cloud, and retrieves similar samples.

# We created a PromptTemplate and defined several examples of temperature conversions. Next, we instantiated the SemanticSimilarityExampleSelector and created a FewShotPromptTemplate with the selector, example_prompt, and appropriate prefix and suffix.

# Using SemanticSimilarityExampleSelector and FewShotPromptTemplate , we enabled the creation of versatile prompts tailored to specific tasks or domains, like temperature conversion in this case. These tools provide a customizable and adaptable solution for generating prompts that can be used with language models to achieve a wide range of tasks.

