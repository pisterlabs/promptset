from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import requests
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import json

load_dotenv()

RESULTS_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()

def web_serach(query: str, num_results: int = RESULTS_PER_QUESTION):
  results = ddg_search.results(query, max_results=num_results)
  return [result["link"] for result in results]

SUMMARY_TEMPLATE = """{text}

---

Using the above text, answer in short the following question:

> {question}

---

if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc.
"""

SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)

def scrape_text(url: str):
  # Send a GET request to the webpage
  response = requests.get(url)

  # Check if the request was successful
  if response.status_code == 200:
    # parse the contents of the webpage with BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # extract all text from the webpage
    text = soup.get_text(separator=" ", strip=True)

    # return the text
    return text
  else:
    # Print error message
    print("Error: Could not retrieve webpage")
    return f"Error: Could not retrieve webpage at {url}"

url = "https://blog.langchain.dev/announcing-langsmith/"

scrape_and_summarize_chain = RunnablePassthrough.assign(
  summary = RunnablePassthrough.assign(
  text=lambda x: scrape_text(x["url"])[:10000]
) | SUMMARY_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()
) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")

web_search_chain = RunnablePassthrough.assign(
  urls = lambda x: web_serach(x["question"])
) | (lambda x: [{"question": x["question"], "url": url} for url in x["urls"]]) | scrape_and_summarize_chain.map()

SEARCH_PROMPT = ChatPromptTemplate.from_messages([
 (
   "user",
   "Write 3 google search queries to search online that form an "
   "objective opinion from the following: {question}\n"
   "You must respond with a list of strings in the following format: "
   '["query 1", "query 2", "query 3"].',
 )
])

search_question_chain = SEARCH_PROMPT | ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0) | StrOutputParser() | json.loads

full_research_chain = search_question_chain | (lambda x: [{"question": q for q in x}]) | web_search_chain.map()

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501

# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""

prompt = ChatPromptTemplate.from_messages(
  [
    ("system", WRITER_SYSTEM_PROMPT),
    ("user", RESEARCH_REPORT_TEMPLATE),
  ]
)

def collapse_list_of_lists(list_of_lists: list[list[str]]):
  content = []
  for l in list_of_lists:
    content.append("\n".join(l))
  return "\n\n".join(content)


report_chain = RunnablePassthrough.assign(
  research_summary= full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI(model="gpt-3.5-turbo-1106") | StrOutputParser()

