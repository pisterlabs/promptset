import logging
import mimetypes
import os
import time
from io import StringIO
from typing import List

import nbformat
import tiktoken
from langchain import callbacks
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langsmith import Client
from nbformat.reader import NotJSONError

from schema import AlttexterResponse, ImageAltText


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def is_valid_notebook(content):
    try:
        nbformat.reads(content, as_version=4)
        return True
    except NotJSONError:
        return False

def remove_outputs_from_notebook(notebook_content):
    if not is_valid_notebook(notebook_content):
        raise ValueError("The content is not a valid Jupyter Notebook")

    notebook = nbformat.reads(notebook_content, as_version=4)

    for cell in notebook.cells:
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None

    logging.info("Successfully removed outputs from the notebook")

    output_stream = StringIO()
    nbformat.write(notebook, output_stream)

    return output_stream.getvalue()

def alttexter(input_text: str, images: dict, image_urls: List[str]) -> List[ImageAltText]:
    """
    Processes input text and images to generate alt text and title attributes.

    Args:
        input_text (str): Article text.
        images (dict): Base64 encoded images.
        image_urls (List[str]): Image URLs.

    Returns:
        Tuple[AlttexterResponse, str]: Generated alt texts and optional tracing URL.
    """
    llm = ChatOpenAI(
        verbose=True,
        temperature=0, 
        model="gpt-4-vision-preview",
        max_tokens=4096
    )

    # For very large text check if it's an ipynb and if so remove the output cells
    if num_tokens_from_string(input_text, "cl100k_base") > 30000:
        if(is_valid_notebook(input_text)):
            input_text = remove_outputs_from_notebook(input_text)

    content = [
        {
            "type": "text",
            "text": f"""ARTICLE:

{input_text}
            """
        }
    ]

    # Process images and add to content
    for image_name, base64_string in images.items():
        mime_type, _ = mimetypes.guess_type(image_name)
        if not mime_type:
            logging.warning(f"Could not determine MIME type for image: {image_name}")
            continue

        image_entry = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_string}",
                "detail": "auto",
            }
        }
        content.append(image_entry)

    # Add image URLs to content
    for url in image_urls:
        image_entry = {
            "type": "image_url",
            "image_url": {
                "url": url,
                "detail": "auto",
            }
        }
        content.append(image_entry)

    parser = PydanticOutputParser(pydantic_object=AlttexterResponse)

    system_prompt = SystemMessagePromptTemplate.from_template(
        template="""You are a world-class expert at generating concise alternative text and title attributes for images defined in technical articles written in markdown format.

For each image in the article use a contextual understanding of the article text and the image itself to generate a concise alternative text and title attribute.
        
{format_instructions}""",
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    all_image_identifiers = list(images.keys()) + image_urls
    messages = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            HumanMessage(content=content),
            HumanMessage(
                content=f"Tip: List of file names of images including their paths or URLs: {str(all_image_identifiers)}"
            ),
        ]
    )

    alttexts = None
    run_url = None

    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    if tracing_enabled:
        client = Client()
        with callbacks.collect_runs() as cb:
            try:
                alttexts = llm.invoke(messages.format_messages()) 
            finally:
                # Ensure that all tracers complete their execution
                wait_for_all_tracers()

            if alttexts:
                try:
                    # Get public URL for run
                    run_id = cb.traced_runs[0].id
                    time.sleep(2)
                    client.share_run(run_id)
                    run_url = client.read_run_shared_link(run_id)
                except Exception as e:
                    logging.error(f"LangSmith API error: {str(e)}")
    else:
        try:
            alttexts = llm(messages.format_messages())
        except Exception as e:
            raise Exception(f"Error during LLM invocation without tracing: {e}")

    alttexts_parsed = parser.parse(alttexts.content) if alttexts else None
    return alttexts_parsed, run_url