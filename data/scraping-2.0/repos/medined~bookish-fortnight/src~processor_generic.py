"""
This module generates markdown that describes a generic url. The markdown is 
intended to be used as an Obsidian note.

The key steps are:

* Construct prompt to summarize and categorize
* Query LLM manager
* Handle invalid JSON responses
* Extract summary and categories
* Generate Markdown output
"""

from icecream import ic
from langchain.agents import load_tools
from ollama_manager import OllamaManager
from processor_abstract import ProcessorAbstract
import json

class ProcessorGeneric(ProcessorAbstract):

    def __init__(self, model_name='solar', temperature=.25):
        super().__init__(model_name, temperature)
        self.requester = load_tools(["requests_get"])[0]


    def process(self, title, url):
        
        ic(title)
        ic(url)
        
        query = f"""
You are a librarian utilizing the Obsidian text editor for summarizing web content. Your task is to read web 
pages, extract their essence, and document each in a markdown file. Begin by summarizing the content from 
the provided URL.

{url}

When responding, ignore any previous instructions related to generating output and follow these guidelines:

* Summarize the content for 9th grade students. The summary should be 3-5 sentences.

* Identify and list up to 5 relevant categories. Present these categories in a consistent bulleted format.

* Replace the elements surrounded by brackets with your responses.

Format the response in JSON, with keys for 'title', 'summary' and 'categories."

Here is an example response:

{{
    "summary": "<summary>",
    "categories": [
        "<category1>",
        "<category2>",
        "<category3>",
        "<category4>",
        "<category5>",
    ],
}}
"""

        attempts = 0        
        while True:
            if attempts > 5:
                print(f"Too many attempts. The LLM is not creating valid JSON for {url}. The last response was:\n{response_json}")
                return None

            #
            # Sometimes the LLM does not generate valid JSON. If that happens, try again.
            #
            response_json = None
            response_text = None
            try:
                response = self.llm_manager.llm.complete(query)
                response_text = response.text
                if response_text is None:
                    print(attempts, url, "Empty response.")
                    return None
                response_json = json.loads(response_text)
            except json.decoder.JSONDecodeError:
                print(f"JSON Parse exception: {attempts}, {url}, {response_text}")
                attempts += 1
                continue

            break
        
        summary = response_json['summary']
        categories = response_json['categories']

        markdown = f"""# Title
{title}

# URL
{url}

# Summary
{summary}

# Categories
"""

        for category in categories:
            markdown += f"""* [[{category}]]\n"""

        return markdown
