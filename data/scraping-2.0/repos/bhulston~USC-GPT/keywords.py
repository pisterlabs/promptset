import openai

def keyword_agent(query):

    system_prompt = """
    You are an AI assistant that looks at a given query about a school class, and comes up with a list of 3 keywords related to the query.
        Two keywords will be used that are similar in semantics to the given query but not already used in query.
        The third keyword should focus on an aspect related to the query that is less obvious, but might be helpful when finding similar data entries.
    If the query does not seem to relate to a school class, please output nothing or just spaces " ".
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    return response["choices"][0]["message"]["content"]