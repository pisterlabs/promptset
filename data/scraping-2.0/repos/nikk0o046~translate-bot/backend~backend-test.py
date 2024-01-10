from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from secret_keys import OPENAI_API_KEY

#input variables
source_language = "English"
target_language = "Finnish"
source_text = """1
00:00:00,000 --> 00:00:05,000
Welcome to the testing session.

2
00:00:05,500 --> 00:00:10,000
This is an example subtitle.

3
00:00:10,500 --> 00:00:15,000
Feel free to customize it as needed.

4
00:00:15,500 --> 00:00:20,000
Enjoy your testing!"""

#initialize the openai model
chat = ChatOpenAI(temperature=0, openai_api_key = OPENAI_API_KEY)

#create the prompt templates
system_template = """You are a helpful assistant that translates {input_language} to {output_language}.
    The content you translate are srt files. Do not attempt to translate it word by word. Rather, be logical and translate the meaning of the text, keeping the time stamps in place.
    Finally, do not include anything other the translated srt file in your response."""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

#request the response from the model
response = chat(
    chat_prompt.format_prompt(
        input_language=source_language, output_language=target_language, text=source_text
    ).to_messages()
)

print(response.content)