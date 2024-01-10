import openai

openai.api_key = 'Your_API_KEY'

query = "how many positions in the kama sutra"

bias = "Only answer questions related to programming.  If the question is not programming specific reply with 'this is not a coding question' and nothing else."

response = openai.ChatCompletion.create(
model="gpt-3.5-turbo",
messages=[
    {"role": "system", "content": ""},
    {"role": "assistant", "content":  bias},
    {"role": "user", "content": query}
]
)
file = open("wiki.html", "a")
answer = (response["choices"][0]["message"]["content"])
input = ("<h1>") + (query) + ("</h1>") + ("<pre>") + (answer) + ("</pre>")
file.write(input)
file.close()
print(query)
print(answer)
