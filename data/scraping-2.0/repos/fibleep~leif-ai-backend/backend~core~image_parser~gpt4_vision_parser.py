import os

from langchain.chains import create_extraction_chain_pydantic
from langchain.chat_models import ChatOpenAI
from openai import OpenAI

from .image_parsing_strategy import ImageParsingStrategy


class GPTParser(ImageParsingStrategy):
    """
    Parse text form an image
    ! IMPORTANT ! GPT Vision doesn't support function calling yet.
    """

    def __init__(self):
        pass

    def parse(self, image, format):
        """
        Parse the image using vision and extract it into a format

        :param image: image to parse, base64
        """

        if format is None:
            prompt = ("Explain what you can see in this image, do not mention "
                      "anything about the words you see in bottom left and bottom "
                      "right:")
            return self.call_vision_api(image, prompt)

        prompt = "Extract the text from this image:"
        # Call the vision API
        text_response = self.call_vision_api(image, prompt)

        # Extract the text from the summary
        llm = ChatOpenAI(
            api_key=os.environ["BACKEND_OPENAI_API_KEY"],
            temperature=0.7,
            model="gpt-4",
        )
        # Create the extraction chain
        extraction_chain = create_extraction_chain_pydantic(
            pydantic_schema=format,
            llm=llm,
        )

        return extraction_chain.run(text_response)

    def call_vision_api(self, image, prompt):
        """
        Call the vision API

        :param image: image to parse, base64
        :param prompt: prompt to use
        """
        client = OpenAI(
            api_key=os.environ["BACKEND_OPENAI_API_KEY"],
        )

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + image},
                        },
                    ],
                },
            ],
            max_tokens=300,
        )
        text_response = response.choices[0].message.content

        return text_response
