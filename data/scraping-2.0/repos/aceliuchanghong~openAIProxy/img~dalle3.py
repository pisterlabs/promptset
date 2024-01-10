import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

proxyHost = "127.0.0.1"
proxyPort = 10809

proxies = {
    "http": f"http://{proxyHost}:{proxyPort}",
    "https": f"http://{proxyHost}:{proxyPort}"
}
openai.proxy = proxies

# Call the API
response = openai.images.generate(
    model="dall-e-3",
    prompt="a cute cat with a hat on",
    size="1024x1024",
    quality="standard",
    n=1,
)

# Show the result that has been pushed to an url
print(response.data[0].url)
