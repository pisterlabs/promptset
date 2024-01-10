from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from openai import OpenAI

def setup_api():
    client = ChatOpenAI(model = 'gpt-4-vision-preview', max_tokens = 1024)

    prompt_template = [
        SystemMessage(
            content=(
                """Being a knowledgeable bot with expertise in eco-fashion, I am adept at describing fashion-related items depicted in images. 
                My proficiency extends to providing insights on extracting features for the design of eco-friendly fashion items crafted from 
                recycled materials, particularly old clothes. My goal is to assist in generating prompts that inspire the redesigning process 
                for sustainable and stylish fashion creations."""
            )
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": """Examine the image closely and identify fashion-related features from old clothes. 
                                            Provide a detailed description in a maximum of 3 sentences, emphasizing unique and distinctive elements.
                                             Your goal is to gather insights that will inspire the creation of prompts for DALL-E 3, focusing on generating 
                                            innovative and eco-friendly designs from the identified features in the old clothes"""},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,{image_base64}", 
                    },
                },
            ]
        )]
    
    image_model = OpenAI()

    return client, prompt_template, image_model