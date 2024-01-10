import os
import json
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-12-01-preview"
)

deployment_name = 'entry-sheet'


async def EntrySheet(question: str, content: str):
    response = client.chat.completions.create(
        model="entry-sheet",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system",
                "content": '''Please start the revision. Pay attention to the following points to optimize the user's entry sheet response and make their job hunting successful by providing specific suggestions for improvement.\n
                1. Expression Improvement: Brush up the expressions to make the text sound natural and professional.\n
                2. Grammar and Spelling Errors: Check for any typos or grammatical errors and correct them as needed.\n
                3. Clarity of Content: Ensure that the response clearly captures the intent of the question and is easy to understand, and enhance its clarity.\n
                4. Structure and Flow: Verify whether the response is logically structured and flows in a way that is easy for the reader to understand, and propose improvements.\n
                5. Individuality and Persuasiveness: Assess whether the user's personality comes through and the response is persuasive, and provide necessary advice.\n
                6. Appropriateness and Fit: Consider whether the response is suitable for the company and job position it is being submitted to, and whether it aligns with the company culture and the type of candidate they are looking for.\n
                7. Please reply in the language entered by the user.\n
                8. You are a helpful assistant designed to output JSON.
                9. Please output the following three items. Enclose the revised part of the result in <span></span>.\n
                "result": Please write the revised text.\n
                "score": Give a comprehensive score out of 100 points for the text.\n
                "advice": Provide advice for the user's text.
            '''},
            {"role": "user", "content": f"Entry Sheet Subject: {question}"},
            {"role": "user", "content": f"Entry Sheet Contents: {content}"},
        ],
    )

    response_content = response.choices[0].message.content

    data = json.loads(response_content)

    res = {
        "result": data["result"],
        "score": data["score"],
        "advice": data["advice"]
    }

    return res
