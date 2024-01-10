from flask import Flask, request, Response
from openai import OpenAI
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente
load_dotenv()

# Inicializar aplicação Flask
app = Flask(__name__)

# Inicializar cliente OpenAI com chave da API do arquivo .env
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@app.route("/")
def hello_world():
    # Obter o texto de entrada do parâmetro de consulta
    input_text = request.args.get('text', '')

    # Gerar fala usando a API da OpenAI
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=input_text
    )

    # Obter dados de áudio usando a propriedade content
    audio_data = response.content

    # Retornar dados de áudio como uma resposta
    return Response(audio_data, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(debug=True)

