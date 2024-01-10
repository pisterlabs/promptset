import os
import openai

openai.api_key = os.getenv(
    "GPT_API")


def summarize(text):
    return openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt="Summarize:\n{text}",
        temperature=0.3,
        max_tokens=4000-len(text),
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )


def sendToGpt(summaryText, jobText):
    if len(summaryText) > 1000:
        summaryText = summarize(summaryText)
    if len(jobText) > 1000:
        jobText = summarize(jobText)
    prom = "Generate a motivation letter from this summary and job-offer:\nSummary:{summaryText}\nJob-Offer:{jobText}"
    return openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prom,
        temperature=0.3,
        max_tokens=4000-len(prom),
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
