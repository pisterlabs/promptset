import openai
import os

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

path = "multimodel_docs/"

# Extract images, tables, and chunk text
from unstructured.partition.pdf import partition_pdf

raw_pdf_elements = partition_pdf(
    filename=path + "1706.03762.pdf",
    extract_images_in_pdf=True,
    extract_tables_in_pdf=True,
    infer_table_structure=True,
    chunking_strategy="by_title",
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
    image_output_dir_path=path,
)

# Categorize text elements by type
tables = []
texts = []
for element in raw_pdf_elements:
    if "unstructured.documents.elements.Table" in str(type(element)):
        tables.append(str(element))
    elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
        texts.append(str(element))

# Multi-modal embeddings with our document

import os

from langchain.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings

# Create chroma
vectorstore = Chroma(
    collection_name="mm_rag_clip_photos",
    embedding_function=OpenCLIPEmbeddings()
)

# Get image URIs with .jpg extension only
image_uris = sorted(
    [
        os.path.join(path, image_name)
        for image_name in os.listdir(path)
        if image_name.endswith(".jpg")
    ]
)

# Add images
vectorstore.add_images(uris=image_uris)
# Add documents
vectorstore.add_texts(texts=texts)
# Make retriever
retriever = vectorstore.as_retriever()

# RAG

import base64
import io

from PIL import Image


def resize_base64_image(base64_string, size=(128, 128)):
    """
        Resize an image encoded as a Base64 string.

        Args:
        base64_string (str): Base64 string of the original image.
        size (tuple): Desired size of the image as (width, height).

        Returns:
        str: Base64 string of the resized image.
        """

    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False


def split_image_text_types(docs):
    """Split numpy array images and texts"""
    images = []
    text = []
    for doc in docs:
        doc = doc.page_content  # Extract Document contents
        if is_base64(doc):
            # Resize image to avoid OAI server error
            images.append(
                resize_base64_image(doc, size=(250, 250))
            )  # base64 encoded str
        else:
            text.append(doc)
    return {"images": images, "texts": text}


from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough


def prompt_func(data_dict):
    # Joining the context texts into a single string
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
            },
        }
        messages.append(image_message)

        # Adding the text message for analysis
        text_message = {
            "type": "text",
            "text": (
                "As an expert in software engineering research papers, your task is to analyze and interpret images, "
                "considering their architectural and technical  significance. Alongside the images, you will be "
                "provided with related text to offer context. Both will be retrieved from a vectorstore based "
                "on user-input keywords. Please use your extensive knowledge and analytical and software engineering "
                "skills to provide a comprehensive summary that includes:\n"
                "- A detailed description of the architectural and technical details in the image.\n"
                "- The technical context of the image.\n"
                "- An interpretation of the image's architecture and its applicability.\n"
                "- Connections between the image and the related text.\n\n"
                f"User-provided keywords: {data_dict['question']}\n\n"
                "Text and / or tables:\n"
                f"{formatted_texts}"
            ),
        }

        messages.append(text_message)
        return [HumanMessage(content=messages)]


model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=2048)

# RAG pipeline
chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(prompt_func)
        | model
        | StrOutputParser()
)

# Test retrieval and run RAG

from IPython.display import HTML, display


def plt_img_base64(img_base64):
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    # Display the image by rendering the HTML
    display(HTML(image_html))


docs = retriever.get_relevant_documents("Transformer architecture", k=3)
for doc in docs:
    if is_base64(doc.page_content):
        plt_img_base64(doc.page_content)
        print("response from the retriever is ", doc.page_content)
    else:
        print(doc.page_content)

response = chain.invoke("Transformer architecture")
print("response from the chain is", response)

