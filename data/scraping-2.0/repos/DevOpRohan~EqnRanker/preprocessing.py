import asyncio
import re
from openAiApi import OpenAIAPI
from utils import remove_false_and_none_lines
from prompts import extract_equation_sys_prompt, extract_equation_prompt, extract_equation_features_sys_prompt, \
    extract_equation_features_prompt, extract_concepts_sys_prompt, extract_concepts_prompt, \
    extract_equation_type_sys_prompt, extract_equation_type_init_prompt


class ContentPreprocessor:
    def __init__(self):
        self.llm = OpenAIAPI()

    async def extract_equation(self, content):
        messages = [
            {
                "role": "system",
                "content": extract_equation_sys_prompt,
            },
            {
                "role": "user",
                "content": extract_equation_prompt.format(problem=content)
            }
        ]
        completion = await self.llm.chat_completion(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )

        equation = completion.choices[0].message["content"]
        # if equation contains "@error", then return None
        if "@error" in equation:
            return None
        else:
            #remove the @eqn: prefix
            equation = re.sub(r'@eqn:', '', equation)
            return equation
        return None
    async def extract_equation_features(self, content):
        messages = [
            {
                "role": "system",
                "content": extract_equation_features_sys_prompt,
            },
            {
                "role": "user",
                "content": extract_equation_features_prompt.format(problem=content)
            }
        ]
        completion = await self.llm.chat_completion(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )

        features = completion.choices[0].message["content"]
        return features

    async def extract_equation_type(self, content):
        messages = [
            {
                "role": "system",
                "content": extract_equation_type_sys_prompt,
            },
            {
                "role": "user",
                "content": extract_equation_type_init_prompt.format(problem=content)
            }
        ]
        completion = await self.llm.chat_completion(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )

        equation_type = completion.choices[0].message["content"]
        return equation_type
    async def extract_concepts(self, content):
        messages = [
            {
                "role": "system",
                "content": extract_concepts_sys_prompt,
            },
            {
                "role": "user",
                "content": extract_concepts_prompt.format(problem=content)
            }
        ]
        completion = await self.llm.chat_completion(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )

        concepts = completion.choices[0].message["content"]
        return concepts

    async def preprocessContent(self, content):
        # Extract the concepts
        concepts = await self.extract_concepts(content)
        # Extract the equations
        equation = await self.extract_equation(content)

        processed_content = ""

        # if concepts isn't None
        if concepts:
            processed_content += "Concepts:\n"
            # Add the concepts to the content
            processed_content += concepts

        # if equation isn't None
        if equation is not None:
            """
            processed_content += "\n\n====\nEquation Features:\n"
            # Extract the features of the equations
            features = await self.extract_equation_features(equation)

            # Experimental: Remove the attributes having false or None values
            # features = remove_false_and_none_lines(features)

            # Add the features to the content
            processed_content += features
            """
            # Add the type of the equation
            processed_content += "\n\n====\nEquation Type:\n"
            equation_type = await self.extract_equation_type(equation)
            processed_content += equation_type

        # add the content to the processed content
        processed_content += "\n\n====\nExample:\n"
        processed_content += content
        return processed_content

    async def preprocessQuery(self, query):
        # Extract the concepts
        concepts = await self.extract_concepts(query)
        # Extract the equations
        equation = await self.extract_equation(query)

        processed_content = ""

        # if concepts isn't None
        if concepts:
            processed_content += "Concepts:\n"
            # Add the concepts to the content
            processed_content += concepts

        # if equation isn't None
        if equation is not None:
            #Rather then feature extraction, Type extraction is more intuitive and token optimized approach in case of semantic search.
            #Feature extraction will be fine in case of normal search instead of semantic search.
            """
            processed_content += "\n\n====\nEquation Features:\n"
            # Extract the features of the equations
            features = await self.extract_equation_features(equation)

            # Experimental: Remove the attributes having false or None values
            features = remove_false_and_none_lines(features)

            # Add the features to the content
            processed_content += features
            """
            # Add the type of the equation
            processed_content += "\n\n====\nEquation Type:\n"
            equation_type = await self.extract_equation_type(equation)
            processed_content += equation_type
        # add the content to the processed content
        processed_content += "\n\n====\nExample:\n"
        processed_content += query
        return processed_content

    async def _preprocess_contents(self, contents):
        tasks = [self.preprocessContent(content) for content in contents]
        preprocessed_contents = await asyncio.gather(*tasks)
        return preprocessed_contents

    def preprocess_text_file(self, file_path, delimiter='\n\n'):
        contents = self.read_text_file(file_path, delimiter)
        loop = asyncio.get_event_loop()
        preprocessed_contents = loop.run_until_complete(self._preprocess_contents(contents))
        return preprocessed_contents

    def read_text_file(self, file_path, delimiter='\n\n'):
        with open(file_path, 'r') as file:
            contents = file.read().split(delimiter)
        return contents


# === TESTS === ##

# # Create an instance of ContentPreprocessor
# content_preprocessor = ContentPreprocessor()
#
# # Specify a single content
# content = 'Solve 2x +3 = 5'

# # Test the extract_concepts function
# concepts = asyncio.run(content_preprocessor.extract_concepts(content))
# print("Concepts:")
# print(concepts)
#
# # Test the extract_equation function
# equation = asyncio.run(content_preprocessor.extract_equation(content))
# print("\nEquation:")
# print(equation)
#
# # Test the extract_equations_features function
# features = asyncio.run(content_preprocessor.extract_equation_features(equation))
# print("\nEquation Features:")
# print(features)
#
# # Test the preprocessContent function
# preprocessed_content = asyncio.run(content_preprocessor.preprocessContent(content))
# print("\nPreprocessed Content:")
# print(preprocessed_content)

# Test the preprocess_text_file function
# preprocessor = ContentPreprocessor()
# preprocessed_contents = preprocessor.preprocess_text_file("data.txt", "====")
# for content in preprocessed_contents:
#     print(content)
