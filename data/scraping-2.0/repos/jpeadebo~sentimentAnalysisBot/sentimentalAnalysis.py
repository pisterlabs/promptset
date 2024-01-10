import websiteAccess
import os
import openai


montleyFoolArticles = websiteAccess.fromListToOpenAIReadable(websiteAccess.getMontleyFool())

openai.api_key = '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'


print("here")
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt="Classify the sentiment in these tweets:\n" + montleyFoolArticles,
    max_tokens=60,
    echo=True,
    temperature=0,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)
print(response, montleyFoolArticles)

