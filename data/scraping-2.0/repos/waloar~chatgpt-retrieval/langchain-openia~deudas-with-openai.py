import sys
import openai
import constants
# os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Configura tu clave de API de OpenAI
openai.api_key = constants.APIKEY

# Preguntas de ejemplo
questions = [
    "¿Cómo puedo obtener un crédito?",
    "Quiero sacar un crédito, ¿cuáles son los requisitos?",
    "Tengo una deuda pendiente, ¿cómo puedo pagarla?",
    "Necesito liquidar mi deuda, ¿cuál es el proceso?",
    "¿Cuáles son las tasas de interés para los créditos?",
    "yo tendría que pagar el 20 una cuota que debo",
    "quiero pagar mi credito",
    "cuanto es lo que debo"
]

# Clasificador ficticio (aquí deberías usar un modelo real o técnica de clasificación)


def classify_intent(response):
    # Simplemente como ejemplo, aquí asumimos que el modelo devuelve "Crédito" o "Deuda"
    return "Crédito" if "crédito" in response.lower() else "Deuda"


# Procesa las preguntas y determina la intención usando GPT-3.5
for question in questions:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=question + "\nRespuesta:",
        max_tokens=50
    )

    answer = response.choices[0].text.strip()
    intent = classify_intent(answer)

    print(
        f"Pregunta: {question}\nRespuesta: {answer}\nTipo de Intención: {intent}\n")

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    query = None
