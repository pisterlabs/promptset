# This module provides necessary backend for the LLM service.

from langchain.llms import VLLM, LlamaCpp
from langchain.prompts.chat import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pydantic import BaseModel
import re

# model_path = "CalvinU/Llama-2-7b-chat-hf-awq"

# # Instantiate the VLLM model for inference
# llm = VLLM(
#     model=model_path,
#     trust_remote_code=True,
#     max_new_tokens=128,
#     top_k=10,
#     top_p=0.95,
#     temperature=0.8,
# )

# Load the LLM model for inference
model_path = "/home/jk249/service/Llama-2-7b-chat-hf/llama-2-7b-chat-hf.gguf.q4_k_m.bin"

llm = LlamaCpp(
    model_path=model_path,
    temperature=1,
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=False,
)

# Topic generation template
topic_gen_template = """
<s>[INST] <<SYS>>
You are a list generator. Your reponse should only contain a list of 5 relevant topics. They should be one or two words in length. Do not include any other information.
<</SYS>>

I am visiting {destination_country} for a vacation. My hobby is {hobby} and my favorite food is {food}. 
I want relevant topics. Example topics include: Greetings, Public utilities, Night life, Shopping, etc. [\INST]
"""

# Topic generation prompt and chain
topic_gen_prompt = ChatPromptTemplate.from_template(topic_gen_template)
topic_gen_chain = topic_gen_prompt | llm

# Phrase generation template
phrase_gen_template = """
<s>[INST] <<SYS>>
You are a list generator. Your reponse should only contain a list of 3 relevant phrases. Do not include any other information.
<</SYS>>

I am visiting {destination_country}. My hobby is {hobby} and my favorite food is {food}. I want relevant phrases about {topic}. [\INST]
"""

# Phrase generation prompt and chain
phrase_gen_prompt = ChatPromptTemplate.from_template(phrase_gen_template)
phrase_gen_chain = phrase_gen_prompt | llm


class PhraseGenerationResponse(BaseModel):
    phrases: list
    translations: list


class TopicGenerationResponse(BaseModel):
    topics: list


# Generate topics
def generate_topics(request) -> TopicGenerationResponse:
    response = topic_gen_chain.invoke(
        {
            "destination_country": request.destination_country,
            "hobby": request.hobby,
            "food": request.food,
        }
    )

    topics = re.findall(r"\d+\.\s(.+)", response)

    return TopicGenerationResponse(topics=topics)


# Generate phrases
def generate_phrases_and_translations(request) -> PhraseGenerationResponse:
    response = phrase_gen_chain.invoke(
        {
            "topic": request.topic,
            "destination_country": request.destination_country,
            "hobby": request.hobby,
            "food": request.food,
        }
    )

    phrases = [
        m
        for match in re.findall(
            r'\d+\.\s*"([^"]*)|\d+\.\s*(¿[^?]*\?)|\d+\.\s*(¡[^?]*!)', response
        )
        for m in match
        if m
    ]

    translations = re.findall(r"\((.*?)\)", response)

    return PhraseGenerationResponse(phrases=phrases, translations=translations)
