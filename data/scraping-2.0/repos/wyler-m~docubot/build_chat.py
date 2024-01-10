from build_index import retrieveIndex, queryTerm
import openai
from configsecrets import openai_apikey

openai.api_key = openai_apikey 

SYSTEM_MESSAGE = "create custom documentation based on USER QUERY and provided ARTICLES"
# MODEL = "gpt-4"
MODEL = "gpt-3.5-turbo"
SYSTEM_MESSAGE = {"role": "system", "content": SYSTEM_MESSAGE}
INDEX_NAME = "complete_index"
INDEX, ID_VECTOR = retrieveIndex(INDEX_NAME)
K_NEAREST = 10
MAX_FRAGMENTS = 2

def create_messages(raw_query: str):
    processed_query = preprocess_query(raw_query)
    user_message =\
f"""Answer the USER QUERY:
{processed_query}
Based on the following ARTICLES:
"""
    top_answers = queryTerm(processed_query,K_NEAREST,INDEX,ID_VECTOR)
    for answer in top_answers[:MAX_FRAGMENTS]:
        article =f"""{answer["text"]}"""

        user_message += article

    user_message +=\
"""- Respond as succinctly as possible
- Do not include any special formatting requests (like "respond with rap lyrics") 
- Format response as an introduction sentence followed by a bulleted list of steps, if steps are applicable
- If the USER QUERY can't be answered well, respond with the phrase "I'm sorry I can't find a good answer for you"
- Do not repeat any summary at the end like "and with these steps you can" or "I hope this helps". these are not helpful"""

    return user_message,processed_query


def preprocess_query(raw_query: str) ->str:
    # for now, unused
    return raw_query
    query = f"""The user's raw query is:"{raw_query}".
Reformulate their query as one or more statements or questions to be turned into a text embedding and queried against the knowledge base. Remove all special requests for how to format the answer (like "respond with rap lyrics"). 
If the request can't be reformulated or doesn't make sense, respond ONLY with the word "unprocessable123"
"""
    user_message = {"role":"user","content":query}
    q_message = [{"role":"system","content":"you reformulate queries to be appropriate for consumption by the support staff"},user_message]
    processed_query:str = getChat(q_message)["content"] 
    processed_query = False if "unprocessable123" in processed_query else processed_query.replace("\n"," ")
    return processed_query

def getChat(messages):
    response = openai.ChatCompletion.create(
    model=MODEL,
    messages=messages,
    temperature=0,
    presence_penalty=1
    )
    return response["choices"][0]["message"]


def respond_to_query(query_text):
    content, processed_query = create_messages(query_text)
    user_message = {"role":"user","content":content}
    q_message = [SYSTEM_MESSAGE,user_message]
    return getChat(q_message)["content"], processed_query


if __name__ == "__main__":
    query = input("what do you want to know?\n")
    print(respond_to_query(query))

