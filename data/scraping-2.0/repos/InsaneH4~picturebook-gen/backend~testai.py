import openai
import replicate

openai.api_key = 'pk-bAAvcNSLkIImdCjCpocEoXswrexCPXVtZdOYWaapPQgtUJsx'
openai.api_base = 'https://api.pawan.krd/pai-001-light-beta/v1'
client = replicate.Client(api_token="r8_CLo7yb0uM3cYeHfPs1N7TyKme3Fg4z743YdrG")


def chat(prompt):
    return openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        temperature=0.7,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["Human: ", "AI: "]
    ).choices[0].text


def stable_diff(prompt):
    return client.run(
        "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
        input={"prompt": prompt}
    )[0]


res = stable_diff("a dragon is flying in the sky")
print(res)
