import modal

stub = modal.Stub(image=modal.Image.debian_slim().pip_install("openai"))


@stub.function(secret=modal.Secret.from_name("my-openai-secret"))
def complete_text(prompt):
    import openai

    completion = openai.Completion.create(model="ada", prompt=prompt)
    return completion.choices[0].text


@stub.local_entrypoint()
def main(prompt: str = "The best way to run Python code in the cloud"):
    completion = complete_text.remote(prompt)
    print(prompt + completion)
