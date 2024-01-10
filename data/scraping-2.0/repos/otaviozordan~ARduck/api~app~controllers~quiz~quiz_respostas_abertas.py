#https://blog.devgenius.io/chatgpt-how-to-use-it-with-python-5d729ac34c0d -> Tutorial

from app import app, db
from app.models.questoes_table import Questoes, verificarRespostaCorreta
from flask import Response, request
import json
import openai

# Define OpenAI API key 
openai.api_key = "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"

# Set up the model and prompt
model_engine = "davinci"

@app.route("/validar_resposta_aberta", methods=["POST"])
def validar_resposta_aberta():
    response = {}
    try:
        body = request.get_json()
        pergunta = body['pergunta']
        resposta = body['resposta']
        
    except Exception as e:
        response = {'Retorno': "Parametros invalidos ou ausentes", 'erro': str(e)}

    try:
        prompt = f"Considere que voce é um professor faz a seguinte pergunta a um de seus alunos:\"" + pergunta + "\" e seu aluno responde:\"" + resposta + "\". Avalie se a resposta está valida com correta ou errado e avalie de 0 a 10 a resposta e caso incorreto diga qual seria a resposta correta."
        # Generate a response
        completion = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        avaliacao = completion.choices[0].text
        response["GPT:"] = avaliacao

    except Exception as e:
        response = {'Retorno': "IA indisponivel", 'erro': str(e)}
    
    return Response(json.dumps(response), status=200, mimetype="application/json")