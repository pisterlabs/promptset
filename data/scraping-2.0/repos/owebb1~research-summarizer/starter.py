import arxiv
import dotenv
import os

from langchain.agents.agent_toolkits import GmailToolkit
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.callbacks import get_openai_callback

dotenv.load_dotenv()

OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")

# Topic of the newsletter you want to write about
query = "LLM"

search = arxiv.Search(
    query=query,
    max_results=4,
    sort_by=arxiv.SortCriterion.SubmittedDate
)


docs = ""

for result in search.results():
    docs += "Title: " + result.title + "\n"
    docs += "Abstract: " + result.summary + "\n"
    docs += "Download URL: " + result.pdf_url + "\n"
    print(result.links)
    for link in result.links:
        docs += "Links: " + link.href + "\n"

# Track cost
with get_openai_callback() as cb:

    # Template for the newsletter
    prompt_newsletter_template = """
    You are a newsletter writer. You write newsletters about scientific articles. You introduce the article and show a small summary to tell the user what the article is about.

    You're main goal is to write a newsletter which contains summaries to interest the user in the articles.

    --------------------
    {text}
    --------------------

    Start with the title of the article. Then, write a small summary of the article.

    Below each summary, include the link to the article containing /abs/ in the URL.

    Summaries:

    """

    PROMPT_NEWSLETTER = PromptTemplate(
        template=prompt_newsletter_template, input_variables=["text"])

    # Initialize the language model
    llm = ChatOpenAI(
        temperature=0.6, model_name="gpt-3.5-turbo-16k", verbose=True)

    # Initialize the LLMChain
    newsletter_chain = LLMChain(
        llm=llm, prompt=PROMPT_NEWSLETTER, verbose=True)

    # Run the LLMChain
    newsletter = newsletter_chain.run(docs)

    # Write newsletter to a text file
    with open("newsletter.txt", "w") as f:
        f.write(newsletter)

    # Set toolkit
    toolkit = GmailToolkit()

    # Initialize the Gmail agent
    agent = initialize_agent(
        tools=toolkit.get_tools(),
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Run the agent
    instructions = f"""
    Send an email to owebb99@gmail.com.
    The subject should be 'Scientific Newsletter about {query}'.
    The content should be the following: {newsletter}.
    """
    agent.run(instructions)
    print(cb)
