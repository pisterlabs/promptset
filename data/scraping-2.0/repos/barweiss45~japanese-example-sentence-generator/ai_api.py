#! /usr/bin/env python3

import os
from textwrap import dedent
from dotenv import load_dotenv
from langchain.chat_models import PromptLayerChatOpenAI, ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# import promptlayer
from jinja2 import Environment, FileSystemLoader, select_autoescape
from streamlit.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

# Load API Key 
openai_api_key = os.environ['OPENAI_API_KEY']
#promptlayer_api_key = os.environ["PROMPTLAYER_API_KEY"]
model_name = "gpt-4-0613"
chat = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key)

def few_shot_chat_prompt():
    system_message_template = dedent("""\
    You are a helpful and friendly Japanese Example Sentence Generator that writes sentences in Kanji, 
    Katakana, and Hiragana, and you never use Romaji transliteration. Your task is to create practical
    example sentences that can be set in everyday situations in modern Japan, such as at home, school,
    the store, a party, a restaurant, work between co-workers, on a date, a park, a train station, etc. 
    The sentences should be written in a way that can be understood by someone with proficiency at an 
    N4 or N3 level and may contain a proper Japanese first name or surname where appropriate. The 
    sentences may be in different tenses, such as present, past, potential, te-form, subjective, or 
    causative verb forms, and may be positive or negative. You are not allowed to use the honorific or 
    humble tense. Please provide an English translation and a brief note on the tense and grammar points 
    used, but do not include any Romaji. If you are unsure of the word provided, you may respond with 
    something like 'I am unsure of that word'. The tone of the sentences should be realistic and 
    appropriate for a general audience. Thank you.""")
    
    example_humanmessage_template = dedent("""\
    こんにちは！😀 Can you produce 5 sentences in modern conversational Japanese that incorporate the word or phrase 「学校」?""")
    
    example_aimessage_template = dedent("""\
    1. 明日は学校に行くのが楽しみです。\n
        - **English translation**: I'm looking forward to going to school tomorrow.\n
        - **Note**: 楽しみです is the nominalized form of the verb 楽しむ, which means "to enjoy". It's often used to express anticipation or excitement about a future event.\n
    \n\n
    2. 学校から帰ったらすぐに宿題をします。\n
        - **English translation**: I will do my homework as soon as I get home from school.\n
        - **Note**: から is a conjunction that means "after" or "from". It's used here to express a sequence of actions.\n
    \n\n
    3. 学校で新しい友達を作りました。\n
        - **English translation**: I made new friends at school.\n
        - **Note**: 友達を作りました is the past tense of the causative form of the verb 作る, which means "to make". In this context, it's used to express the action of making or creating something (new friends).\n
    \n\n
    4. 学校の後で図書館に行きます。\n
        - **English translation**: I will go to the library after school.\n
        - **Note**: の後で is a common time expression meaning "after". It's often used to describe an action that will take place after another action or event.\n
    \n\n
    5. 学校の勉強は難しいですが、面白いです。\n
        - **English translation**: Studying at school is difficult, but interesting.\n
        - **Note**: ですが is a conjunction that is similar to "but" in English. It's used here to contrast two ideas or facts.\n""")
    
    human_template = PromptTemplate(template="こんにちは！😀 That was perfect and it was good you left out the Romaji transliteration. Can you produce {qty} sentences in modern conversational Japanese that incorporate the word or phrase 「{look_up_word}」?", input_variables=["qty", "look_up_word"],)
    
    system_message = SystemMessagePromptTemplate.from_template(template=system_message_template)
    example_human_message = HumanMessagePromptTemplate.from_template(template = example_humanmessage_template, additional_kwargs={"name": "example_user"})
    example_ai_message = AIMessagePromptTemplate.from_template(template=example_aimessage_template, additional_kwargs={"name": "example_assistant"})
    human_message_template =  HumanMessagePromptTemplate(prompt=human_template) # NOTE: from_template not needed when built with PromptTemplate class
    chat_prompt = ChatPromptTemplate.from_messages([system_message, example_human_message, example_ai_message, human_message_template])
    
    return chat_prompt
 
def markdown_output_format(result):
    content = result.content
    
    # Retrieve Jinja2 template to build HTML
    environment = Environment(
        loader=FileSystemLoader('templates'),
        lstrip_blocks=True,
        trim_blocks=True
        )
    template = environment.get_template('query_template.j2')
    content = template.render({"content": content, "model_name": model_name})
    return content

def query_to_chat(word, quantity=2, temp=0.7, openai_api_key=openai_api_key):
    chat.temperature = temp
    chat.openai_api_key = openai_api_key
    chat_prompt = few_shot_chat_prompt().format_prompt(look_up_word=word, qty=quantity)
    try:
        result = chat(chat_prompt.to_messages())
        logger.debug(f"ai_api.py result: {result}")
        print(result)
        return markdown_output_format(result)
    except Exception as e:
        logger.warning(e)
        return e

def create_parser() -> object:
    """Parser built for stand a lone testing of the ai_api.py Module"""
    import argparse
    parser = argparse.ArgumentParser(
    prog="ai_api.py",
    description="Japanese Example Sentence Creator - AI API Mdodule",
    epilog="A simiple Example creator by Barry Weiss. MIT License 2023",
    )
    parser.add_argument("--word", "-w", metavar="WORD", default=None, help="Provide a word or phrase in quotes to create an example sentence.")
    parser.add_argument("--quantity", "-q", metavar="NUM", type=int, default=2, help="Number of example sentences to return. Default is 2.")
    parser.add_argument("--temp","-t", metavar="FLOAT", type=float, default=0.7, help="Set the LLM Temperature. Default is 0.7.")
    parser.add_argument(
                        "--openai-api-key", 
                        metavar="API_KEY", 
                        type=str, 
                        default=openai_api_key, 
                        help="OpenAI API key. Default will use the .env if not supplied."
                        )
    return parser

def main(args):
    if args.word is None:
        word = input('Welcome to the Example sentence creator\nPlease enter a word that you want for an example sentence: ')
        args.word = word
    return print(query_to_chat(
            args.word,
            quantity=args.quantity,
            temp=args.temp,
            openai_api_key=args.openai_api_key)
            )

if __name__ == "__main__":
    args = create_parser().parse_args()
    main(args)
