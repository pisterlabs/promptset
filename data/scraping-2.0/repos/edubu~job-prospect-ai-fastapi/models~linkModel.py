from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate

from config import OPENAI_API_KEY

# output parser
output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

# prompts
LINK_PROMPT = """\
Below are links that were found on the main page of {url}:
----------------------------------------
{links}
----------------------------------------
Please choose a few links that you think are most important to discover more about what the company does. Include a few that give information on their products, along with a few that tell more about the company.
For examples:
- I may choose 4 links that have /solutions in the title so that I can learn more about their products.
- I may choose 2 links that have /about or /who-we-are in the title so that I can learn more about the company.
- I may choose 2 links that have /careers in the title so that I can learn more about the company culture.
- I will always choose a link that has /pricing in the title so that I can learn more about the affordability of their products

Give a list of only the ones that seem most important. Return a maximum of ${num_links} links.
{format_instructions}
"""

unused_lines_in_prompt = """
I want to avoid choosing links that seem to be about a product that is not relevant to what the website is about. For example,
If the website is an ecommerce store, I do not want to scrape product descriptions as they do not give much information about the website
"""

# prompt templates
link_template = PromptTemplate(
    input_variables=["url", "links", "num_links"],
    template=LINK_PROMPT,
    partial_variables={"format_instructions": format_instructions},
)

# LlMs
llm = OpenAI(
    openai_api_key=OPENAI_API_KEY, model_name="text-davinci-003", temperature=0
)

# Chains
link_chain = LLMChain(llm=llm, prompt=link_template, output_parser=output_parser)
