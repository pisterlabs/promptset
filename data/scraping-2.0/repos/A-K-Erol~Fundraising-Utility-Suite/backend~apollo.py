import requests
from openai import OpenAI

url = "https://api.apollo.io/v1/people/match"

data = {
    "api_key": "",
    "first_name": "Mathias",
    "last_name": "Wilder",
    "organization_name": "Deep Instinct",

    "domain": "deepinstinct.com",
    "reveal_personal_emails": True,
    "linkedin_url": "https://www.linkedin.com/in/mathias-widler/"
}

headers = {
    'Cache-Control': 'no-cache',
    'Content-Type': 'application/json'
}

client = OpenAI()
raw2 = requests.request("POST", url, headers=headers, json=data)

# print(raw.text)
# response = client.chat.completions.create(
#         model="gpt-3.5-turbo-1106",
#         messages=[
#             {"role": "system", "content": "You are a lead generation and client management expert who synthesizes / "
#                                           "json information into appealing text-based displays"},
#             {"role": "user", "content": str(raw.text)}
#         ]
#     )
response2 = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": "You are a lead generation and client management expert who synthesizes / "
                                          "json information into appealing displays. Write me an html element for this."},
            {"role": "user", "content": str(raw2.text)}
        ]
    )


print(response2.choices[0].message.content)
