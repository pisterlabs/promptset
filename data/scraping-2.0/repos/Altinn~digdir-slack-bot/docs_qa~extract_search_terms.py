import os
import openai
import instructor
from pydantic import BaseModel, Field
import pprint

instructor.patch()
openai.api_type = 'azure'
openai.api_key = os.environ['OPENAI_API_KEY_ALTINN3_DEV']
openai.api_base = os.environ['OPENAI_API_URL_ALTINN3_DEV']
openai.api_version = os.environ['AZURE_OPENAI_VERSION']


class GeneratedSearchQueries(BaseModel):
    searchQueries: list[str] = Field(..., description="Array of search queries.")
    userInputLanguage: str = Field(..., description="ISO 639-1 language code for the original question")

pp = pprint.PrettyPrinter(indent=2)


async def run_query_async(user_input) -> GeneratedSearchQueries:    
    query_result: GeneratedSearchQueries = openai.ChatCompletion.create(
        engine=os.environ['AZURE_OPENAI_DEPLOYMENT'],
        response_model=GeneratedSearchQueries,
        temperature=0.1,
        messages=[
            {"role": "system", 
             "content": f"""You have access to a search API that returns relevant documentation.

1. If the question is not in English, first translate to English.             

2. Generate the shortest set of search terms that sufficiently limits the set of expected query results.

3. Generate an array of up to 7 search queries that are relevant to this question. Use a variation of related keywords and synonyms for the queries, trying to be as general as possible.
Include as many queries as you can think of, including and excluding terms.
For example, include queries like ['keyword_1 keyword_2', 'keyword_1', 'keyword_2'].
Be creative. The more queries you include, the more likely you are to find relevant results."""},
            {"role": "user", "content": user_input},
        ]
    )

    for i in range(len(query_result.searchQueries)):
        query_result.searchQueries[i] = query_result.searchQueries[i].replace("GitHub", "").strip()
        

    return query_result