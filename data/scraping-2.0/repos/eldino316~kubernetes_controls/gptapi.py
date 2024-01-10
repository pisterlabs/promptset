import openai


def creation_template():
    openai.api_key = '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'
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Donnez-moi un template de déploiement dans un fichier .yml d'un nginx sur kubernetes.",
        max_tokens=300
    )

    return response

res = creation_template()
print(res.choices[0].text.strip())