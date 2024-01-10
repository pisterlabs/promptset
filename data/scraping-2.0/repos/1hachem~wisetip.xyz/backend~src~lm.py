import os

import dotenv
import openai
from bardapi import Bard
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from src.utils.utils import parse, text_split

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model_name="text-davinci-003")
chat_llm = ChatOpenAI(temperature=0)


class Chain:
    def __init__(self) -> None:
        pass

    def chatgpt_inference(self, transcript: str) -> list[str]:
        """chatgpt inference, not working yet"""
        system_template = """
        You are a helpful assistant that Summarize this text from the user into a list of tips. 
        The tips are clear bullet points extracted from the text.
        Avoid mentioning any advertising or promotional language. 
        start each tip with a star *
        dont come up with tips that are not mentioned in the text, 
        if there is no tip output (# no tip found)        
        """
        system_message_prompt = SystemMessage(content=system_template)
        chunks = text_split(transcript)
        batch_messages = []
        for text in chunks:
            batch_messages.append(
                [
                    system_message_prompt,
                    HumanMessage(content="text:" + text + "tips :"),
                ]
            )

        response = chat_llm.generate(batch_messages)
        output = "  ".join([batch[0].text for batch in response.generations])
        output = parse(output)

        return output

    def gpt3_inference(self, transcript: str) -> list[str]:
        prompt_template = """
        Summarize this text into tips 
        The tips are clear bullet points extracted from the text.
        Avoid mentioning any advertising or promotional language. 
        start each point with a star *
        dont come up with tips that are not mentioned in the text
        text : {text}
        tips : 
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=prompt,
            combine_prompt=prompt,
        )

        chunks = text_split(transcript)
        docs = [Document(page_content=c) for c in chunks]

        output = chain({"input_documents": docs})["output_text"]
        output = parse(output)
        return output

    def bard_summary(self, transcript: str) -> list[str]:
        prompt_template = """
        summarize this transcript : {text}
        summary :
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

        bard_token = os.getenv("_BARD_API_KEY")
        bard = Bard()

        outputs = []
        chunks = text_split(transcript)
        for text in chunks:
            request = prompt.format(text=text)
            output = bard.get_answer(request)["content"]
            outputs.append(output)

        return outputs
