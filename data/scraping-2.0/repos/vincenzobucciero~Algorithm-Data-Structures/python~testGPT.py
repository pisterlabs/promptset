import openai

# Imposta la tua chiave API
openai.api_key = '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'

# Fai una domanda al modello di linguaggio GPT-3.5
domanda = "Qual è il significato della vita?"
risposta = openai.Completion.create(
    engine="text-davinci-003",  # Specifica il motore del modello (in questo caso, GPT-3.5)
    prompt=domanda,  # La domanda che vuoi porre al modello
    max_tokens=150  # Il numero massimo di token nella risposta generata
)

print(risposta.choices[0].text.strip())  # Stampa la risposta generata dal modello
