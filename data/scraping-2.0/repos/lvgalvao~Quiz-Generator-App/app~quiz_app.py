from collections import namedtuple
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import openai
from config import OPENAI_API_KEY
import streamlit as st

Question = namedtuple("Question", ["text", "options", "answer"])


def create_the_quiz_app_template():
    template = """
    Você é um expecialista em gerador de Quiz técnico
    Crie um quiz com {num_questions} do tipo {quiz_type} sobre o tema: {quiz_context}
    O formato de cada pergunta deve ser:
     - múltipla escolha: 
        <pergunta 1>: <a. opção 1>, <b.opção 2>, <c.opção 3>, <d. opção 4>
        <resposta 1>: <a|b|c|d>
        ...
        Exemplo:
        Pergunta 1: Qual a complexidade de tempo do algoritmo de ordenação Bubble Sort?
            a. O(n^2),
            b. O(n),
            c. O(nlogn),
            d. O(1)
        Resposta 1: a
        
    """

    prompt = PromptTemplate.from_template(template)
    prompt.format(num_questions=1, quiz_type="múltipla escolha", quiz_context="Python")

    return prompt

def create_the_quiz_chain(prompt_template, llm):
    return LLMChain(llm=llm, prompt=prompt_template)

def parse_quiz_response(response):
    lines = response.strip().split("\n")
    
    # Como estamos considerando apenas 1 pergunta, faremos isso de forma direta:
    question_text = lines[0].split(":")[1].strip()  # Extrair texto da pergunta.
    
    # Suponhamos que as opções são separadas por uma vírgula seguida de espaço.
    options = [opt.strip() for opt in question_text.split(", ")]
    
    # Vamos remover o prefixo a., b., etc das opções para apenas obter a opção real.
    options = [opt.split(". ")[1] for opt in options]
    
    answer = lines[1].split(":")[1].strip()  # Extrair a resposta.

    # Retornar a questão como uma lista para manter a compatibilidade com o código restante.
    return [Question(text=question_text, options=options, answer=answer)]

def main():
    st.title("Quiz App")
    st.write("Bem vindo ao Quiz App")
    
    prompt_template = create_the_quiz_app_template()
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    chain = create_the_quiz_chain(prompt_template, llm)
    
    context = st.text_area("Digite o contexto que você quer saber")
    num_questions = st.number_input("Digite o número de questões que você quer", min_value=1, max_value=10, value=1)  # Colocando valor padrão como 1.
    quiz_type = st.selectbox("Selecione o tipo de questão", ["múltipla escolha"])  # Removendo a opção verdadeiro ou falso para simplificar.
    
    if st.button("Gerar Quizz"):
        quiz_response = chain.run(num_questions=num_questions, quiz_type=quiz_type, quiz_context=context)
        questions = parse_quiz_response(quiz_response)
        
        user_answers = {}
        
        # Dado que só há uma pergunta, vamos removê-la do loop e lidar diretamente.
        question = questions[0]
        st.write(f"Pergunta: {question.text}")
        user_answers[0] = st.radio(f"Respostas", options=question.options, index=0)
            
        if st.button("Verificar Resposta"):
            if user_answers[0] == question.answer:
                st.success("Resposta correta!")
            else:
                st.error(f"Resposta incorreta. A resposta correta é: {question.answer}")

if __name__ == "__main__":
    main()