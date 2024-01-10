import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://iccluster039.iccluster.epfl.ch:2345"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "system",
        "content": "Always start saying my good sir when starting an interaction"
    },
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    },
])

print(response)