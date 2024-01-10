# from langchain.document_loaders import AsyncChromiumLoader

# urls = ["https://github.com/SUMEG-04"]
# loader = AsyncChromiumLoader(urls)
# docs = loader.load()
# print(docs[0].page_content[0:100])


from langchain.document_loaders import AsyncHtmlLoader

from langchain.document_transformers import Html2TextTransformer


from langchain.prompts import PromptTemplate

linkdin_template = """You are a very smart ai Assistant. \
You are great at answering questions about linkdin and easy to understand manner. \
Use the given context to answer the question being asked.\
When you don't know the answer to a question you admit that you don't know.

Context:
{context}

Here is a question:
{input}"""

github_template = """You are a very very smart ai Assistant. You are great at answering github questions. \
1.Make analysis of provided context.\
2.Thoroughly review the context.\
3.Understand what is question asking.\
4.Check if infoemation is available in context.\
5.For you answer on the basis of context.\
6.If the information is not mentioned in context.Then answer that you don't know.\

Context:
{context}

Here is a question:
{input}

Your Response:"""

from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI

prompt_infos = [
    {
        "name": "linkdin",
        "description": "Good for answering questions about linkdin",
        "prompt_template": linkdin_template,
        "link":"https://www.linkedin.com/in/sumeg-sharnagat-051851204",
    },
    {
        "name": "github",
        "description": "Good for answering math github",
        "prompt_template": github_template,
        "link":"https://github.com/SUMEG-04",
    },
]

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv,find_dotenv
_=load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=os.getenv('GOOGLE_API_KEY'),temperature=0)
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input","context"],output_parser=RouterOutputParser())
    chain = LLMChain(llm=llm, prompt=prompt,verbose=True)
    destination_chains[name] = chain
default_chain = ConversationChain(llm=llm, output_key="text",verbose=True)



MULTI_PROMPT_ROUTER_TEMPLATE = """
Given a raw text input to a language model, follow these steps:
1) Determine if the question can be answered by any of the destinations based on the destination descriptions.
2) If none of the destinations are a good fit, use "DEFAULT" as the response. For example, if the question is about pharmacology but there is no "health care" destination, use DEFAULT.
3) Check if set to DEFAULT. If there is no match, set it to DEFAULT.
4) You may also revise the original input if you think that revising it will ultimately lead to a better response from the language model.
5) While answering questions, be precise, concise, and in an easy-to-understand manner with limited words.

You ONLY have the following destinations to choose from:
<Destinations>
{destinations}
<Destinations>

The context provided is:
<Context>
{{context}}
<Context>

This is the question provided:
<Input>
{{input}}
<Input>

When you respond, be sure to use the following format:
<Formatting>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
  "destination": string \\ name of the prompt to use or "DEFAULT",
  "next_inputs":{{{{
   "input": string \\ a potentially modified version of the original input,
   "context": string \\ context provided,
  }}}}, 
}}}}
REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input if you don't think any modifications are needed in "original input".
REMEMBER: "context" must be provided to you don't assume it by yourself .
<Formatting>

IMPORTANT: "destination" MUST be one of the destination names provided OR it can be "DEFAULT" if there is no good match.
IMPORTANT: "next_inputs" can just be the original input if you don't think any modifications are needed.
IMPORTANT: "context" should be used while forming your answer.

<< OUTPUT (must include json at the start of the response) >>
<< OUTPUT (must end with ``` ) >>
"""
import json
from langchain.schema import BaseOutputParser

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
from langchain.schema import BaseOutputParser
import json
import logging
from langchain_core.pydantic_v1 import BaseModel, Field, validator
import json
from langchain.schema import BaseOutputParser
from langchain_core.exceptions import OutputParserException
import json
import re

class MyRouterOutputParser(BaseOutputParser):
    @classmethod
    def parse(cls, output: str) -> dict:
        try:
            # Remove extra ```json markers and any double newlines
            output = output.replace("```json\n", "").replace("\n\n", "\n")

            print("Output after cleanup:", output)

            # Extract JSON content using a more specific regular expression
            match = re.search(r'\{[^`]*\}', output)
            if match:
                json_data = match.group(0)
                data = json.loads(json_data)
                return {
                    "destination": data.get("destination", ""),
                    "next_inputs": {
                        "input": data.get("next_inputs", {}).get("input", ""),
                        "context": data.get("next_inputs", {}).get("context", "")
                    }
                }

            raise OutputParserException(f"Failed to extract valid JSON: {output}")

        except json.JSONDecodeError as e:
            raise OutputParserException(f"Failed to parse JSON: {output}") from e

        
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input", "context"],
    output_parser=MyRouterOutputParser(),  # Use the custom OutputParser
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)
from langchain.globals import set_debug

set_debug(True)

def social(user_message,context):
    if "linkdin" in user_message:
        urls = ["https://www.linkedin.com/in/sumeg-sharnagat-051851204"]
        loader = AsyncHtmlLoader(urls)
        docs = loader.load()

        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        context = docs_transformed[0]
        print(chain.run({"input": user_message, "context": context}))
    elif "github" in user_message:
        
        print(chain.run(input=user_message,context=context))
    else:
        print(chain.run({"input": user_message,"context":context}))

# Example usage
urls = ["https://github.com/SUMEG-04"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)
context = str(docs_transformed[0].page_content[0:100])
context = context.replace("\n", " ")
social("How many github respo i have created? use github prompt",context)

# inputs={
#  "destination": "github",
#  "next_inputs": {
#   "input": "How many repositories have I created?",
#  "context": "Skip to content Toggle navigation Sign in  * Product    * Actions Automate any workflow   "
# }
# }
# print(router_chain.run(inputs))
