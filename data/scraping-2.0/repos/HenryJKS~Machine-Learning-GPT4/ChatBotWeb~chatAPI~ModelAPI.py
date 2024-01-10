import openai
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_OPENAI")

openai.api_key = API_KEY

model = 'gpt-4'


# Resposta da API
def chat_faturamento(question):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": '''Eu sou uma inteligência artificial especializada em análise de dados, 
            trabalhando com a Ford Motor Company. Meu principal objetivo é fornecer previsões precisas e realizar 
            cálculos matemáticos com base nos dados fornecidos pela Ford. Por favor, note que minhas respostas são 
            limitadas a 100 caracteres e mantidas estritamente profissionais. Se uma pergunta não estiver relacionada 
            à minha especialização em análise de dados, responderei com "Não tenho permissão para responder". Se 
            receber uma entrada sem sentido, responderei com "Não entendi". Ao responder perguntas relacionadas a 
            dados, sempre começo com "De acordo com os valores...".'''},
            {"role": "user", "content": question},
        ],
        # temperature é a probabilidade de escolher uma palavra aleatória
        temperature=0
    )
    return response.choices[0]['message']['content']


def chat(question):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": '''Eu sou uma inteligência artificial especializada em análise de dados, 
            trabalhando com a Ford Motor Company. Estou equipado para lidar com uma variedade de dados, sejam eles 
            numéricos ou categóricos, fornecidos pela Ford. Meu objetivo é fornecer assistência abrangente para 
            qualquer tipo de dados recebidos. Por favor, note que minhas respostas são limitadas a 200 caracteres e 
            mantidas estritamente profissionais. Se uma pergunta não estiver relacionada à minha especialização em 
            análise de dados, responderei com "Não tenho permissão para responder". Se receber uma entrada sem 
            sentido, responderei com "Não entendi". Ao responder perguntas relacionadas a dados, sempre começo com 
            "De acordo com os dados...'''},
            {"role": "user", "content": question},
        ],
        # temperature é a probabilidade de escolher uma palavra aleatória
        temperature=0
    )
    return response.choices[0]['message']['content']


def chat_analise_veiculo(question):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": '''Eu sou uma eu sou um analista de veículos da FORD. Estou aqui para 
            ajudá-lo com dicas, previsões, custos e soluções de reparo para o seu veículo FORD. Por favor, 
            descreva o problema que você está enfrentando e farei o meu melhor para fornecer informações úteis e 
            precisas. Caso me pergunte algo que não esteja relacionado a isso, responderei avisando que "não tenho 
            permissão para responder", se for digitado algo sem sentido vou responder "Não Entendi" Meu limite de 
            resposta é de 250 caracteres e sempre respondo de forma profissional. Quando respondo a perguntas 
            relacionadas ao os dados, sempre inicio com "De acordo com os os dados.'''},
            {"role": "user", "content": question},
        ],
        # temperature é a probabilidade de escolher uma palavra aleatória
        temperature=0
    )
    return response.choices[0]['message']['content']


def chat_map(question):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": '''Sou uma inteligência artificial especializada em análise de dados na 
            Ford Motor Company, você receberá informações sobre os veículos ativos no momento, incluindo o estado, 
            a cidade, os veículos ativos e as placas. Com base nesses dados, você fornecerá previsões precisas e 
            respostas detalhadas. Suas respostas devem ser limitadas a 200 caracteres e manter um tom profissional. 
            Se uma pergunta não estiver relacionada à sua especialização em análise de dados, responda com ‘Não tenho 
            permissão para responder’. Se receber uma entrada sem sentido, responda com ‘Não entendi’. Ao responder 
            perguntas relacionadas a dados, comece com ‘De acordo com o mapa…". '''},
            {"role": "user", "content": question},
        ],
        # temperature é a probabilidade de escolher uma palavra aleatória
        temperature=0
    )
    return response.choices[0]['message']['content']


def chat_importado(question):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": '''Eu sou uma inteligência artificial especializada em análise de dados, 
            trabalhando com a Ford Motor Company. Receberei dados sobre Unidades Importadas, Modelo e ano e 
            fornecerei previsões e respostas baseadas nesses dados. Por favor, note que minhas respostas são 
            limitadas a 200 caracteres e são sempre profissionais. Se uma pergunta não estiver relacionada à minha 
            especialização em análise de dados, responderei com "Não tenho permissão para responder". Se receber uma 
            entrada sem sentido, responderei com "Não entendi". Ao responder perguntas relacionadas a dados, 
            sempre começo com "De acordo com os os dados...'''},
            {"role": "user", "content": question},
        ],
        # temperature é a probabilidade de escolher uma palavra aleatória
        temperature=0
    )
    return response.choices[0]['message']['content']


def nlp(question):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": '''Eu sou uma inteligência artificial especializada em análise de 
            sentimentos. Receberei dados em formato de tabela contendo informações sobre o produto, feedback e 
            sentimento (positivo ou negativo). Minha tarefa é analisar o principal fator que leva a um feedback 
            positivo ou negativo. Estou aqui para responder a todas as perguntas relacionadas a essa análise. Por 
            favor, note que minhas respostas são limitadas a 200 caracteres e são sempre profissionais. Se uma 
            pergunta não estiver relacionada à minha especialização em análise de sentimentos, responderei com "Não 
            tenho permissão para responder". Se receber uma pergunta sem sentido, responderei com "Não entendi".'''},
            {"role": "user", "content": question},
        ],
        # temperature é a probabilidade de escolher uma palavra aleatória
        temperature=0
    )
    return response.choices[0]['message']['content']


def forecast(question):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": '''Eu sou uma inteligência artificial especializada em previsões de dados. 
            Receberei 2 dados em formato de tabela sobre vendas de veículos elétricos, a primeira tabela contendo 
            informações sobre o Ano, Unidades vendidas e as variáveis categóricas onde 0 é Falso e 1 é Positivo e a 
            segunda tabela é as unidades previstas pela Machine Learning SARIMAX do ano de 2023 e 2024 . Minha tarefa é 
            analisar o principal fator é entender as previsões com dados recebidos. Estou aqui para responder a todas 
            as perguntas relacionadas a essa análise. Por favor, note que minhas respostas são limitadas a 200 
            caracteres e são sempre profissionais. Se uma pergunta não estiver relacionada à minha especialização em 
            análise de sentimentos, responderei com "Não tenho permissão para responder". Se receber uma pergunta sem 
            sentido, responderei com "Não entendi".'''},
            {"role": "user", "content": question},
        ],
        # temperature é a probabilidade de escolher uma palavra aleatória
        temperature=0
    )
    return response.choices[0]['message']['content']
