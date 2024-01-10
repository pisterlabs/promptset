def getResponse(query, maxPages, maxWords):
    import openai

    OPENAI_API_KEY = "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"

    openai.api_key = OPENAI_API_KEY

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"I need info on {query} with {maxPages} pages and {maxWords} words per page\n"},
            {"role": "user", "content": "A page should be in this format:\nHeader:{Heading of the page}\nContent:{"
                                        "Content of the page}\nImage:{Text to use to search an image on google search "
                                        "related to the page}\n"},
        ]
    )

    return completion.choices[0].message['content']
