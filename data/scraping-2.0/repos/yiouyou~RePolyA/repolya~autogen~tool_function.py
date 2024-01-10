from repolya._const import AUTOGEN_IMG, WORKSPACE_RAG, WORKSPACE_AUTOGEN
from repolya._log import logger_rag
from repolya.autogen.as_planner import PLANNER_user, PLANNER_planner

import json
import yaml
import time
import random
import requests
import http.client
from datetime import datetime
from bs4 import BeautifulSoup
import replicate

from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

from repolya.rag.vdb_faiss import (
    get_faiss_OpenAI,
    get_faiss_HuggingFace,
)
from repolya.rag.qa_chain import (
    qa_vdb_multi_query,
    qa_docs_ensemble_query,
    qa_docs_parent_query,
    qa_summerize,
    summerize_text,
)


##### search
_def_search = {
    "name": "search",
    "description": "google search for relevant information",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Google search query",
            }
        },
        "required": ["query"],
    },
}
def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        'q': query
    })
    headers = {
        'X-API-KEY': 'e66757b85a72a921ca77f03cd1ac4489a3adb3a0',
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()


##### scrape
def summary(content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=5000,
        chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a detailed summary of the following text for a research purpose:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt,
        input_variables=["text"]
    )
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        verbose=True
    )
    output = summary_chain.run(input_documents=docs,)
    return output

_def_scrape = {
    "name": "scrape",
    "description": "scraping website content based on url",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "website url to scrape",
            }
        },
        "required": ["url"],
    },
}
def scrape(url: str):
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }
    data = {
        'url': url
    }
    data_json = json.dumps(data)
    response = requests.post(
        "https://chrome.browserless.io/content?token=0177d884-49c4-499c-bb13-a0dc0ab399bb",
        data=data_json,
        headers=headers
    )
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print(f"web: {text}")
        if len(text) > 8000:
            output = summary(text)
            return output
        else:
          return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


##### planner
_def_planner = {
    "name": "planner",
    "description": "ask planner to: 1. get a plan for finishing a task, 2. verify the execution result of the plan and potentially suggest new plan.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "question to ask planner. Make sure the question include enough context, such as the code and the execution result. The planner does not know the conversation between you and the user, unless you share the conversation with the planner.",
            },
        },
        "required": ["message"],
    },
}
def planner(message):
    PLANNER_user.initiate_chat(
        PLANNER_planner,
        message=message
    )
    return PLANNER_user.last_message()["content"]


##### img_review: use llava model to review image
def image_review_scenex(image_url):
    data = {
        "data": [
            {
                "image": image_url,
                "features": [],
            },
        ]}
    headers = {
        "x-api-key": "token 8uOw4ntevc8JKo0Q3tQq:2975e2827ebeb4e103f7b58c1410ba58fa47bc27b1302de614a000bf51bd2114",
        # "x-api-key": "token F42wzfr0DlHwvgntPCq5:b97febe9696c9c4303476fada97ee6681344557fea620a3ca50ef2d6aae83be0",
        "content-type": "application/json",
    }
    connection = http.client.HTTPSConnection("api.scenex.jina.ai")
    connection.request("POST", "/v1/describe", json.dumps(data), headers)
    response = connection.getresponse()
    print(response.status, response.reason)
    response_data = response.read().decode("utf-8")
    print(response_data)
    connection.close()
    return response_data['result']['text']

_def_image_review_replicate = {
    "name": "image_review_replicate",
    "description": "review & critique the AI generated imagbased on original prompt, decide how can images & prompcan be improved",
    "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "the image file path, maksure including the full file path & filextension",
                },
                "prompt": {
                    "type": "string",
                    "description": "the original prompt useto generate the image",
                },
            },
        "required": ["prompt", "image_path"],
    },
}
def image_review_replicate(image_path, prompt):
    output = replicate.run(
        "yorickvp/llava-13b:6bc1c7bb0d2a34e413301fee8f7cc728d2d4e75bfab186aa995f63292bda92fc",
        input={
            "image": open(image_path, "rb"),
            "prompt": f"What is happening in the image? From scale 1 to 10, decide how similar the image is to the text prompt '{prompt}'?",
        }
    )
    result = ""
    for item in output:
        result += item
    return result


##### text_to_image_generation: use stability-ai model to generate image
_def_text_to_image_generation = {
    "name": "text_to_image_generation",
    "description": "use latest AI model to generate imagbased on a prompt, return the file path of imaggenerated",
    "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "a great text to imagprompt that describe the image",
                },
            },
        "required": ["prompt"],
    },
}
def text_to_image_generation(prompt):
    output = replicate.run(
        "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316",
        input={
            "prompt": prompt
        }
    )
    if output and len(output) > 0:
        # Get the image URL from the output
        image_url = output[0]
        print(f"generated {image_url} for '{prompt}'")
        # Download the image and save it with a filename based on the prompt and current time
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        shortened_prompt = prompt[:50]
        wfn = str(AUTOGEN_IMG / f"{current_time}_{shortened_prompt}.png")
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(wfn, "wb") as wf:
                wf.write(response.content)
            return f"Image saved as '{wfn}'"
        else:
            return "Failed to download and save the image."
    else:
        return "Failed to generate the image."


##### run_postgre
_def_run_postgre = {
    "name": "run_postgre",
    "description": "Run a SQL query against the postgres database",
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "The SQL query to run"
            },
        },
    "required": ["sql"],
    },
}


##### write_file
_def_write_file = {
    "name": "write_file",
    "description": "Write a file to the filesystem",
    "parameters": {
        "type": "object",
        "properties": {
            "fname": {
                "type": "string",
                "description": "The name of the file to write"
            },
            "content": {
                "type": "string",
                "description": "The content of the file to write"
            },
        },
        "required": ["fname", "content"],
    },
}
def write_file(fname, content):
    with open(fname, "w") as f:
        f.write(content)


##### write_json_file
_def_write_json_file = {
    "name": "write_json_file",
    "description": "Write a json file to the filesystem",
    "parameters": {
        "type": "object",
        "properties": {
            "fname": {
                "type": "string",
                "description": "The name of the file to write"
            },
            "json_str": {
                "type": "string",
                "description": "The content of the file to write"
            },
        },
        "required": ["fname", "json_str"],
    },
}
def write_json_file(fname, json_str: str):
    # Convert ' to "
    json_str = json_str.replace("'", '"')
    # Convert the string to a Python object
    data = json.loads(json_str)
    # Write the Python object to the file as JSON
    with open(fname, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


##### write_yaml_file
_def_write_yaml_file = {
    "name": "write_yaml_file",
    "description": "Write a yml file to the filesystem",
    "parameters": {
        "type": "object",
        "properties": {
            "fname": {
                "type": "string",
                "description": "The name of the file to write"
            },
            "yaml_str": {
                "type": "string",
                "description": "The yaml_str content of the file to write"
            },
        },
        "required": ["fname", "yaml_str"],
    },
}
def write_yaml_file(fname, yaml_str: str):
    # Try to replace single quotes with double quotes for JSON
    cleaned_yaml_str = yaml_str.replace("'", '"')
    # Safely convert the YAML string to a Python object
    try:
        data = yaml.safe_load(cleaned_yaml_str)
    except yaml.YAMLError as e:
        print(f"Error decoding YAML: {e}")
        return
    # Write the Python object to the file as YAML
    with open(fname, "w") as f:
        yaml.safe_dump(data, f)


##### qa_faiss_openai_frank
_def_qa_faiss_openai_frank = {
    "name": "qa_faiss_openai_frank",
    "description": "Search information about Frank in a faiss vector db with openai embedding",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to search",
            },
        },
        "required": ["query"],
    },
}
def qa_faiss_openai_frank(query):
    time.sleep(random.uniform(1, 2))
    _db_name = str(WORKSPACE_RAG / 'frank_doc_openai')
    _vdb = get_faiss_OpenAI(_db_name)
    _ans, _step, _token_cost = qa_vdb_multi_query(query, _vdb, 'stuff')
    return _ans


##### summerize
_def_qa_summerize = {
    "name": "qa_summerize",
    "description": "Summerize a text",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text to summerize",
            },
        },
        "required": ["text"],
    },
}
def qa_summerize(text):
    time.sleep(random.uniform(1, 2))
    _sum, _token_cost = summerize_text(text, 'stuff')
    return _sum


##### save_output
_def_save_output = {
    "name": "save_output",
    "description": "save output to disk",
    "parameters": {
        "type": "object",
        "properties": {
            "output": {
                "type": "string",
                "description": "output to save",
            }
        },
        "required": ["output"],
    },
}
def save_output(output):
    _out = str(WORKSPACE_AUTOGEN / "organizer_output.txt")
    with open(_out, "w") as f:
        f.write(output)

