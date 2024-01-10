from article_extractor import get_article
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from api_key import get_api_key
from Output_Parser import get_parser

OPENAI_API_KEY = get_api_key("../../.env")

## getting an Article 
article_url = "https://edition.cnn.com/2023/11/03/investing/who-is-sam-bankman-fried-ftx-fraud-trial/index.html"
article = get_article(article_url)

article_title = article.title
article_text = article.text

## Getting the parser

parser = get_parser()

template = """
As an advanced AI who specialises in summarizing news articles, you've been tasked to summarize online news articles into bulleted points. 

Here's the article you need to summarize:

==================
Title: {article_title}

{article_text}
==================
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables= ["article_title", "article_test"],
    partial_variables= {"format_instructions": parser.get_format_instructions()}    
)

# format the prompt with the input variables
formatted_prompt = prompt.format_prompt(article_title=article_title, article_text=article_text) 

# Instatiate the LLM
model = OpenAI(model_name="text-davinci-003", temperature=0.1, openai_api_key=OPENAI_API_KEY)

# Output
output = model(formatted_prompt.to_string())

# Parse the output
parsed_output = parser.parse(output)
print(parsed_output)