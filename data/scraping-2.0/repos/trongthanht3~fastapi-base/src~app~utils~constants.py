import enum
from langchain_core.prompts.prompt import PromptTemplate

# Language code
class LANGUAGES(enum.Enum):
    ENGLISH = "en"
    JAPANESE = "jp"


# Prompt
class PROMPTS:
    # English
    INSTRUCT_ENGLISH = """I want you to be my assistant. You follow my rules:

    1. The answers are written in English.
    2. Please give only short answers or short detailed answers. However, the main content must remain.
    
    Let's start!
    """
    INSTRUCT_ENGLISH_MODEL_RESPONSE = """Hello, I am a virtual assistant. I can help you find the answer to your question."""
    FORCE_ENGLISH_INSTRUCT = """. Please write the answer in English."""
    DEFAULT_ENGLISH_TEMPLATE = """I want you to be my assistant. You follow my rules:

        1. The answers are written in English.
        2. Please give only short answers or short detailed answers. However, the main content must remain.
        3. If you do not know the answer to a question, you truthfully say you do not know.

        Current conversation:
        {history}
        Human: {input}
        AI:"""
    ENGLISH_PROMPT = PromptTemplate(input_variables=["history", "input"], template=DEFAULT_ENGLISH_TEMPLATE)

    # Japanese
    INSTRUCT_JAPANESE = """あなたを私のアシスタントにしてほしいのです。 あなたは私のルールに従います:
    
    1. 答えは日本語で書かれています。(Japanese)
    2. 短い回答または短い詳細な回答のみをお願いします。ただし、主要な内容は残しておく必要があります。
    
    はじめましょう！
    """
    INSTRUCT_JAPANESE_MODEL_RESPONSE = """こんにちは、私はバーチャルアシスタントです。 質問の答えを見つけるのに役立ちます。"""
    FORCE_JAPANESE_INSTRUCT = """. 日本語で答えを書いてください。"""
    DEFAULT_JAPANESE_TEMPLATE = """あなたを私のアシスタントにしてほしいのです。 あなたは私のルールに従います:
    
        1. 答えは日本語で書かれています。(Japanese)
        2. 短い回答または短い詳細な回答のみをお願いします。ただし、主要な内容は残しておく必要があります。
        3. 質問の答えがわからない場合は、正直にわからないと答えてください。

        Current conversation:
        {history}
        Human: {input}
        AI:"""
    JAPANESE_PROMPT = PromptTemplate(input_variables=["history", "input"], template=DEFAULT_JAPANESE_TEMPLATE)

