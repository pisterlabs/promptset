"""
Summarize.py
-------------
Summarizes the created course
"""

import openai

openai.api_key = "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"


def Create_Summary(paragraphs):
    short_paragraphs = []

    for paragraph in paragraphs:
        prompt = "Summarize and shorten " + paragraph
        short_paragraph = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
            top_p=0.9)
        short_paragraphs.append(short_paragraph["choices"][0]["message"]["content"].strip())

    summary = '\n'.join(short_paragraphs)
    return summary