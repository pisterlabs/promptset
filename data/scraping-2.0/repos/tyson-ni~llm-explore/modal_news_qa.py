import modal
import openai 
import json
import requests
import os
from numpy import dot

stub = modal.Stub(name='news-api-qa', image=modal.Image.debian_slim(python_version='3.10').pip_install(['openai', 'requests', 'numpy']))

GPT_MODEL = "gpt-4-1106-preview"

# Helper functions
@stub.function(secret=modal.Secret.from_name("my-openai-key"))
def json_gpt(prompt: str):

    completion = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "Output only valid JSON"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        request_timeout=60,
        response_format={'type': "json_object"}
    )

    text = completion.choices[0].message.content
    parsed = json.loads(text)

    return parsed


@stub.function()
def generate_news_queries(question: str):

    QUERIES_INPUT = f"""
    You have access to a search API that returns recent news articles.
    Generate an array of search queries that are relevant to this question.
    Use a wide variation of related keywords for the queries, trying to be as general as possible.
    Include as many queries as you can think of, including and excluding terms.
    For example, include queries like ['keyword_1 keyword_2', 'keyword_1', 'keyword_2'].
    Be creative. 
    Please be concise and include only the keywords in the queries. 
    Come up with at least 3 queries and at most 6 queries.

    User question: {question}

    Format: {{"queries": ["query_1", "query_2", "query_3"]}}
    """

    queries = json_gpt.remote(QUERIES_INPUT)["queries"]

    return queries


@stub.function(secret=modal.Secret.from_name("my-openai-key"))
def embeddings(input: list[str]) -> list[list[str]]:
    response = openai.Embedding.create(model="text-embedding-ada-002", input=input)
    return [data.embedding for data in response.data]


@stub.function(secret=modal.Secret.from_name("my-newsapi-key"))
def search_news(query: str, from_datetime: str,  to_datetime: str, num_articles: int = 50) -> dict:

    response = requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": query,
            "apiKey": os.environ['NEWS_API_KEY'],
            "pageSize": num_articles,
            "sortBy": "relevancy",
            "from": from_datetime,
            "to": to_datetime,
        },
    )

    return response.json()


@stub.function()
def query_articles(queries: list[str], from_datetime: str, to_datetime: str):

    articles = []

    for query in queries:
        result = search_news.remote(query, from_datetime, to_datetime)
        if result["status"] == "ok":
            articles = articles + result["articles"]
        else:
            raise Exception(result["message"])

    # remove duplicates
    articles = list({article["url"]: article for article in articles}.values())

    print("Total number of articles: ", len(articles))
    
    return articles


@stub.function()
def rank_articles(articles: list[dict], question: str, top_n: int):

    HA_INPUT = f"""
    Generate a hypothetical answer to the user's question. This answer which will be used to rank search results. 
    Pretend you have all the information you need to answer, but don't use any actual facts. Instead, use placeholders
    like NAME did something, or NAME said something at PLACE. Also pretend you are in the year 2030 and have access to all the knowledge up to that year. 

    User question: {question}

    Format: {{"hypotheticalAnswer": "hypothetical answer text"}}
    """ 

    hypothetical_answer = json_gpt.remote(HA_INPUT)["hypotheticalAnswer"]

    print("Hypothetical answer {}".format(hypothetical_answer))

    hypothetical_answer_embedding = embeddings.remote(hypothetical_answer)[0]
    article_embeddings = embeddings.remote(
        [
            f"{article['title']} {article['description']} {article['content'][0:100]}"
            for article in articles
        ]
    )

    # Calculate cosine similarity
    cosine_similarities = []
    for article_embedding in article_embeddings:
        cosine_similarities.append(dot(hypothetical_answer_embedding, article_embedding))

    scored_articles = zip(articles, cosine_similarities)

    # Sort articles by cosine similarity
    sorted_articles = sorted(scored_articles, key=lambda x: x[1], reverse=True)

    # # Print top 5 articles
    # print("Top 5 articles:", "\n")

    # for article, score in sorted_articles[0:5]:
    #     print("Title:", article["title"])
    #     print("Description:", article["description"])
    #     print("Content:", article["content"][0:100] + "...")
    #     print("Score:", score)
    #     print()


    formatted_top_results = [
        {
            "title": article["title"],
            "description": article["description"],
            "url": article["url"],
        }
        for article, _score in sorted_articles[0:top_n]
    ]

    return formatted_top_results


@stub.function(secret=modal.Secret.from_name("my-openai-key"))
def answer_question(top_articles: list[dict], question: str):

    ANSWER_INPUT = f"""
    Generate an answer to the user's question based on the given search results. 
    TOP_RESULTS: {top_articles}
    USER_QUESTION: {question}

    Include as much information as possible in the answer.
    The summary should be at most three paragraphs long.
    Please reference the relevant search result urls as markdown links at the end of the answer as appendices.
    Your answer should be structured sequentially with the summary followed by all the sources together at the end.

    -- Example Answer --
    There have been several significant scientific and engineering discoveries recently. In the field of artificial intelligence (AI), new tools are being developed rapidly across various sciences, accelerating the 
    pace of breakthroughs in areas such as protein discovery and battery technology  In quantum computing, Atom Computing has announced a record-breaking 1,225-qubit
    quantum computer.

    In the field of neuroscience, scientists have built the largest-ever map of the human brain, which could help 
    explain abilities like language and vulnerabilities like Alzheimer's disease  In the realm of space technology, there have been breakthroughs in satellite 
    tech that harness the sun's power Lastly, in the field of nanotechnology, the 2023 Nobel Prize for chemistry 
    recognized the power of this technology 

    Sources:
    [source 
    1](https://www.npr.org/sections/health-shots/2023/10/12/1205201928/artificial-intelligence-ai-scientific-disco
    veries-proteins-drugs-solar). [source 
    2](https://www.forbes.com/sites/moorinsights/2023/10/24/atom-computing-announces-record-breaking-1225-qubit-qu
    antum-computer/). 
    [source 
    3](https://www.npr.org/sections/health-shots/2023/10/16/1205780690/largest-ever-map-human-brain-atlas-3000-cel
    ls-alzheimers-schizophrenia).[source 
    4](https://www.nzherald.co.nz/business/beaming-solar-energy-from-space-its-not-as-far-fetched-as-it-sounds/Y7U
    YTNQPPZG6XPLSZ3NR22E6MY/).  [source 
    5](https://phys.org/news/2023-10-nobel-prize-chemistry-power-nanotechnology.html).

    ----

    """

    completion = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": ANSWER_INPUT}],
        temperature=0.1,
        request_timeout=90
    )

    return completion.choices[0].message.content


@stub.local_entrypoint()
def main(question: str, from_datetime: str, to_datetime: str, top_n: int = 20):
    queries = generate_news_queries.remote(question)
    articles = query_articles.remote(queries, from_datetime, to_datetime)
    ranked_articles = rank_articles.remote(articles, question, top_n)
    print(answer_question.remote(ranked_articles, question))
    return 
