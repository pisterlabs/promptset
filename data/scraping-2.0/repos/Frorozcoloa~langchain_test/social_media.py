from openai import OpenAI
import base64
from pathlib import Path
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI

from tqdm import tqdm
from typing import List, Dict

from config import openai_api_key, codegpt_api_key, code_gpt_agent_id, codegpt_api_base
from utils import text2json, save_csv

model = OpenAI(api_key=openai_api_key)

def image_b64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def look(image_path, prompt="Describe this image"):
    b64_image = image_b64(image_path)

    response = model.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{b64_image}",
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ],
        max_tokens=1024,
    )

    message = response.choices[0].message
    return message.content

def read_all_images():
    images_paths = Path("images").iterdir()
    description = {}
    for image_path in tqdm(images_paths):
        if image_path.is_dir():
            read_images = image_path.glob("*.jpg")
            for image in read_images:
                describe = look(image)
                doc = Document(page_content=describe, metadata={"source": image_path.name})
                if description.get(image_path.name) is None:
                    description[image_path.name] = [doc]
                
                else:
                    description[image_path.name].append(doc)
    return description


def get_tamplate():
    template = "You are a helpful assistant. Your task is to analyze to draw common topic from the given descriptions of the users"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = """
        Please, identify the main topics mentioned in the description of the users.

        Return a list of 3-5 topics. 
        Output is a JSON list with the following format
        [
            {{"topic_name": "<topic1>", "topic_description": "<topic_description1>"}}, 
            {{"topic_name": "<topic2>", "topic_description": "<topic_description2>"}},
            ...
        ]
        user_1:
        {user_1_description}
        user_2:
        {user_2_description}
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    return chat_prompt


def get_model() -> ChatOpenAI:
    # Create a ChatOpenAI object with the retrieved API key, API base URL, and agent ID
    llm = ChatOpenAI(
        openai_api_key=codegpt_api_key,
        openai_api_base=codegpt_api_base,
        model=code_gpt_agent_id,
    )
    return llm


# Create a list of messages to send to the ChatOpenAI object


def run(users_description: Dict[str,Document]) -> List[Dict]:
    """Returns a list of topics, given a description of a product"""
    llm = get_model()
    chat_prompt = get_tamplate()
    messages = chat_prompt.format_prompt(user_1_description = users_description["user_1"], user_2_description = users_description["user_2"])
    response = llm(messages.to_messages())
    list_desc = text2json(response.content)
    return list_desc
    



if __name__ == "__main__":
    description = read_all_images()
    topics = run(description)
    save_csv(topics, "Anaslysis_social_meida.csv")