import time
from openai import AzureOpenAI

# Set your OpenAI API key
api_key = '264baae4c11546cea94468cf39fe3e76'
client = AzureOpenAI(
    api_version="2023-07-01-preview",
    azure_endpoint="https://oai-int-azg-we-001.openai.azure.com/",
    azure_deployment="dep-gtp-4-32k",
    api_key=api_key
)
data = open('cissp.txt').read().split('========================')


def generate_article(prompt):
    response = client.chat.completions.create(
        model="gpt-4-32k",
        messages=[
            {"role": "system", "content": "Turn this content to bullet point; ensure you are summarizing and not"
                                          " missing important information since it will be used to study for CISSP"
                                          " exam as a preparation. Sound friendly, in basic words, SEO friendly and "
                                          "more like human all the time."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=16 * 1024
    )

    return response.choices[0].message.content


def main():
    n = 1
    for prompt in data:
        while True:
            try:
                print(f"trying {n}")
                f = open("cissp-bulletpoints.md", 'a')
                answer = generate_article(prompt)
                f.write(answer + '\n========================\n')
                f.close()
                n += 1
                break
            except Exception as e:
                print(f"error while requesting {n}, error: {e}")
                time.sleep(1)
    print("done")


if __name__ == "__main__":
    main()
