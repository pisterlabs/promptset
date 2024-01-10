from langchain.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from utils import get_links

# from langchain import
from langchain.output_parsers import CommaSeparatedListOutputParser


LINK_SUGGESTOR_PROMPT = """
You are going to pick a link to visit in order to answer a query. Pick the link that will give you the highest
probability of finding the most relevant information related to the query.

== Query ==
{query}

== Links ==
{links}

{format_instructions}
"""

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
prompt_template = PromptTemplate(
    template=LINK_SUGGESTOR_PROMPT,
    input_variables=["query", "links"],
    partial_variables={"format_instructions": format_instructions},
)


class LinksSuggestor:
    def __init__(self, db: Chroma, links_cache: set):
        self.db = db
        self.links_cache = links_cache
        self.llm = ChatOpenAI(model="gpt-4")

    def suggest_links(self, query: str):
        """
        Suggest links to explore to answer a query

        1. Top k similarity search, k = 1
        2. Get links on page
        3. Suggest links with llm
        """
        retriever = self.db.as_retriever()
        best_doc = retriever.get_relevant_documents(query, k=1)[0]
        url = best_doc.metadata["source"]

        links = set(get_links(url))
        unvisited_links = list(links - (self.links_cache & links))
        response = self.llm(
            [
                HumanMessage(
                    content=prompt_template.format(query=query, links=unvisited_links)
                )
            ]
        )

        links = output_parser.parse(response.content)
        return links[0]  # suggested link
