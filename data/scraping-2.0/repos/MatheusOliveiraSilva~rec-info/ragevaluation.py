from ragas.metrics.critique import harmfulness
from ragas.llama_index import evaluate
import openai
import os
from dotenv import load_dotenv
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    harmfulness,
]

eval_questions = [
    # Pergunta 1
    "O que é inpainting?",
    #Pergunta 2
    "Explique como funciona a parte de contração e expansão da U-Net.",
    #Pergunta 3
    "Explique a operação de MaxPooling2D.",
    #Pergunta 4
    "Explique a operação de Conv2D.",
    #Pergunta 5
    "Fale sobre o conjunto de treinamento usado no artigo.",
]

eval_answers = [
    # Resposta 1
    "O inpainting, técnica que visa restaurar ou preencher regiões corrompidas ou ausentes em imagens.",
    #Resposta 2
    "Na parte de contração, a imagem é progressivamente reduzida em dimensão espacial enquanto aumenta em profundidade."
    "Isso permite que a rede capture características mais abstratas e de alto nível sobre a entrada, como a presença de"
    "formas complexas ou objetos inteiros. No entanto, com essa abstração, perde-se detalhes espaciais finos."
    "Na parte de expansão, a imagem é progressivamente reconstruída para sua dimensão espacial original. Aqui, as "
    "conexões de salto desempenham um papel crucial. Elas rein- troduzem os detalhes espaciais perdidos durante a "
    "codificação, transferindo informações diretamente de uma camada da parte de contração para sua camada "
    "correspondente na parte de expansão. Isso permite que a rede combine informações contextuais de alto nível com "
    "detalhes espaciais de baixo nível, tornando-a extremamente eficaz para segmentação.",
    #Resposta 3
    "A operação MaxPooling2D pode ser visualizada como uma janela deslizante que per- corre a imagem: em cada posição "
    "da janela, apenas o valor máximo dentro da janela é retido.",
    #Resposta 4
    "A operação Conv2D é uma operação de convolução bidimensional que produz um mapa de características 2D.",
    #Resposta 5
    "O conjunto de dados utilizado para treinar e avaliar a U-Net adaptada para inpainting foi o ImageNet "
    "(DENG et al., 2009), uma das bases de dados mais extensas e reconhe- cidas na área de visão computacional."
    "O ImageNet é composto por milhões de imagens coloridas, variando em tamanho, mas muitas delas são redimensionadas "
    "para dimensões comuns, como 256x256 pixels, para facilitar o processamento. A diversidade no tamanho e na natureza"
    " colorida das imagens torna este conjunto de dados ideal para treinar redes neurais em tarefas complexas de "
    "processamento de imagem.",
]

eval_answers = [[a] for a in eval_answers]

def rag_evaluation(query_engine):
    # usa o método de avaliação do framework ragas
    result = evaluate(query_engine, metrics, eval_questions, eval_answers)

    # scores para cada métrica
    print(result)

