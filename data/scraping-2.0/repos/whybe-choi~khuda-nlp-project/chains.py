import os

from dotenv import load_dotenv
from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate

load_dotenv()


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYZE_QUERY_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt_templates", "analyze_query.txt")
CONSULT_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt_templates", "consult.txt")
REVIEW_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt_templates", "review.txt")
EXTRACT_KEYWORDS_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "prompt_templates", "extract_keywords.txt")


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )

llm = ChatOpenAI(temperature=0.1, max_tokens=2048, model="gpt-4")

analyze_query_chain = create_chain(
    llm=llm,
    template_path=ANALYZE_QUERY_PROMPT_TEMPLATE,
    output_key="output",
)
consult_chain = create_chain(
    llm=llm,
    template_path=CONSULT_PROMPT_TEMPLATE,
    output_key="output",
)

review_chain = create_chain(
    llm=llm,
    template_path=REVIEW_PROMPT_TEMPLATE,
    output_key="output"
)

extract_keywords_chain = create_chain(
    llm=llm,
    template_path=EXTRACT_KEYWORDS_PROMPT_TEMPLATE,
    output_key="output"
)

default_chain = ConversationChain(llm=llm, output_key="output")