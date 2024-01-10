import os # Esta biblioteca é usada para acessar variáveis de ambiente
from dotenv import load_dotenv # é usada para carregar variáveis de ambiente a partir de um arquivo
import openai # É a biblioteca da OpenAI que permite acessar os serviços

load_dotenv() # Esta função é usada para carregar as variáveis de ambiente de um arquivo .env na memória
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Geradores são funções cuja execução pode ser interrompida e posteriormente reconduzida.
def generate_response(question):
    openai_stream = openai.ChatCompletion.create( # Aqui, a função create da biblioteca OpenAI é usada para iniciar uma conversa com o modelo de linguagem
            model="gpt-3.5-turbo", # Especifica o modelo de linguagem a ser usado.
            messages=[{"role": "user", "content": f"{question}"}], # Uma lista que contém um dicionário com o papel do remetente ("user" neste caso) e o conteúdo da mensagem (a pergunta).
            temperature=0.0, # A temperatura determina as saídas em uma escala entre 0 e 1; quanto mais próxima de 1, mais aleatória, e quanto mais próxima de 0, mais determinística.
            stream=True, # Habilita o modo de transmissão para permitir a obtenção de respostas progressivas.
        )
    # O código entra em um laço for para iterar sobre as respostas geradas pela API.
    # A cada iteração, verifica se a resposta contém algum conteúdo.
    for line in openai_stream: 
            if "content" in line["choices"][0].delta:
                current_response = line["choices"][0].delta.content
                yield current_response

# yield é usada em funções funções geradoras para criar uma sequência de valores 
# que podem ser recuperados um de cada vez. Quando se usa yield em uma função, 
# ela pausa a execução e retorna um valor. Quando você solicita o próximo valor, 
# a execução é retomada a partir desse ponto.