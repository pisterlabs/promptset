# Text Translation

# It is one of the great attributes of Large Language models that enables them to perform multiple tasks just by changing the prompt. We use the same llm variable as defined before. However, pass a different prompt that asks for translating the query from a source_language to the target_language.

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0
)


translation_template = "Translate the following text from {source_language} to {target_language}: {text}"
translation_prompt = PromptTemplate(
    input_variables=["source_language", "target_language", "text"], 
    template=translation_template
    )
translation_chain = LLMChain(llm=llm, prompt=translation_prompt)


source_language = "English"
target_language = "French"
text = "Your text here"
translated_text = translation_chain.predict(source_language=source_language, target_language=target_language, text=text)
