import csv
import os

from openai import OpenAI

api_key = "OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI()


# Dados tabelados com textos em pt e br
dados = [
    {"Texto_es": "Hola, ¿cómo estás?", "Texto_pt": "Olá, como está?"},
    {
        "Texto_es": "El clima está agradable hoy.",
        "Texto_pt": "O clima está agradável hoje.",
    },
    {
        "Texto_es": "Me encanta la comida española. Especialmente las tapas y paellas.",
        "Texto_pt": "Adoro comida espanhola. Especialmente tapas e paellas.",
    },
    {
        "Texto_es": "Viajar es una experiencia enriquecedora. Nos permite descubrir nuevas culturas y ampliar nuestra perspectiva sobre el mundo.",
        "Texto_pt": "Viajar é uma experiência enriquecedora. Permite-nos descobrir novas culturas e ampliar nossa perspectiva sobre o mundo.",
    },
]

prompts = [
    "Traduza: '{}'",
    "Traduza a frase '{}' para o Português.",
    "Por favor, traduza a frase '{}' para o Português, mantendo a estrutura gramatical e considerando a adequação cultural.",
    "Traduza '{}' para o Português, mantendo a maior fidelidade de sentido entre a origem e sua tradução. Este texto faz parte de um projeto de pesquisa e a tradução precisa ser com a melhor qualidade possível.",
]


def get_translation(prompt, text):
    messages = [{"role": "system", "content": prompt.format(text)}]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


# Dicionário para armazenar os resultados
resultados = {}

# Teste de todas as permutações possíveis
for dado in dados:
    resultados[dado["Texto_es"]] = {}
    for prompt in prompts:
        resultados[dado["Texto_es"]][prompt] = get_translation(prompt, dado["Texto_es"])

# Verifica se o arquivo existe, se não existir, cria um novo
if not os.path.exists("resultados-traducoes.csv"):
    open("resultados-traducoes.csv", "w").close()


# Salvando os resultados em um arquivo CSV
with open(
    "resultados-traducoes.csv", mode="w", encoding="utf-8", newline=""
) as csv_file:
    fieldnames = ["Texto_es", "Texto_pt", "Prompt", "Tradução"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for dado in dados:
        for prompt in prompts:
            writer.writerow(
                {
                    "Texto_es": dado["Texto_es"],
                    "Texto_pt": dado["Texto_pt"],
                    "Prompt": prompt,
                    "Tradução": resultados[dado["Texto_es"]][prompt],
                }
            )
