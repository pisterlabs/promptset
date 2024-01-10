from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
from langchain.llms import OpenAI
from googleapiclient.discovery import build 
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate, LLMChain
import qdran_query as QD
import intent_identi as int_iden
import learningSaveFunction3 as saveProgress
from dotenv import load_dotenv


load_dotenv("/home/kap/Desktop/pythonGPT/keys.env")

materia = "historia e geografia"
llm=OpenAI(model = "gpt-3.5-turbo", temperature=0)


Memoria = []   #array responsavel pela memoria do chat, composto por dicionarios cada um com 2 keys e 2 values  {"user": input, "assistant": text}

linkList =[]  #lista de links para a busca na web

total_tokens, completion_tokens,prompt_tokens, total_cost = 0, 0, 0, 0


materia= f"você é um tutor que explica de forma didática a matéria {materia} "

template0 = """de forma precisa e profunda para os alunos. Pense passo-a-passo no melhor jeito de responder uma pergunta ou resolver algo e apenas traga resposta corretas

{chat_history}
aluno: {human_input}
tutor:"""   #prompt do chat principal


class GoogleSearch():
    def __init__(self):
        self.service = build(
            "customsearch", "v1", developerKey=os.environ.get("GOOGLE_API_KEY")
        )

    def search(self, query,num_results=4):
        response = (
            self.service.cse()
            .list(
                q=query,
                cx=os.environ.get("GOOGLE_CSE_ID"),
                num=num_results
            )
            .execute()
        )
        return response['items']
    def filter(self, response):
        response_with_links = ''

        for result in response:
            response_with_links += f"Link: {result['link']}\n"
            linkList.append(result['link'])

            response_with_links += f"titulo: {result['title']}\n"
            response_with_links += f"Conteúdo: {result['snippet']}\n\n"
        return response_with_links

def classifica(UserInput: str):
        input_parsed= UserInput.lower()
        resultado = int_iden.intent_identi(input_parsed) #problemas no NLU com o modelo não reconhecendo web e internet como keywords de pesquisa_web, aparentemente ele ve as 2 palavras como parte de pesquisa_web mas ele ainda sim vai para chat_padrão 
        print("resultado da ferramenta: " + resultado)

        if  resultado == "pesquisa_web": return 1
        elif resultado == "busca_questoes": return 2
        elif  resultado == "verifica_resposta": return 3
        else: return 0
       
class chatBot():
    def __init__(self, input: str):
        self.input = input 
        self.classificar = classifica(input)

    def chatModel(self):
        input = self.input
        global total_cost, total_tokens, prompt_tokens, completion_tokens
        
        if self.classificar == 1:
             pesquisa = GoogleSearch()
             template = """Você é um tutor conversando com um Aluno.

              Dadas as seguintes partes extraídas de uma pesquisa e uma pergunta do aluno.
              Analise todas as informações da pesquisa, pense passo a passo para dar a resposta mais correta e atualizada possível. De preferencia para informações mais recentes relevantes para o ano de 2023
              SEMPRE  Traga pelo menos 1 link relevante à pergunta/conversa para o aluno, escreva os links para o usuário apenas se ele for diretamente relacionado com o assunto da conversa.
             {context}

             {chat_history}
             aluno: {human_input}
             tutor:""" 

             prompt = PromptTemplate(
                 input_variables=["chat_history", "human_input", "context"], 
                 template=template
             )
             
             llm_chain0 = LLMChain(
                         llm=ChatOpenAI(model = "gpt-3.5-turbo", temperature=0), 
                         prompt=prompt, 
                         verbose=False, 
                         
                         )

             query = input 
             semfiltro= pesquisa.search(query)
             resposta_filtrada= pesquisa.filter(semfiltro)
             messages = ["user: {} \n assistant: {}\n".format(dic['user'], dic['assistant']) for dic in Memoria]
             output = "".join(messages)
             with get_openai_callback() as cb:
                chain = llm_chain0({"human_input" : input, "context": resposta_filtrada, "chat_history": output }, return_only_outputs=True)
              
                total_cost +=cb.total_cost
                total_tokens +=cb.total_tokens
                prompt_tokens +=cb.prompt_tokens
                completion_tokens +=cb.completion_tokens
               
                # print(f"Total Tokens web: {cb.total_tokens}")
                # print(f"Prompt Tokens web: {cb.prompt_tokens}")
                # print(f"Completion Tokens web: {cb.completion_tokens}")
                # print(f"Total Cost (USD) web: ${cb.total_cost}")
             
            
             #print("input_string", chain)
             #print(chain['text'])
             #potencial de erros aqui
             text = chain['text'] 
             retu_dict = {"user": input, "assistant": text}         
             Memoria.append({"user": input, "assistant": text})

             return retu_dict
             

        elif self.classificar == 2:  # chain de pesquisa em docs
            memoria_docs, retornoDOC =  QD.Pesquisa_Questoes(input)
            Memoria.append({"user": input, "assistant": memoria_docs[-1]["context"]})
            retu_dict = {"user": input, "assistant": retornoDOC + "\n\n"}
           #print("retorno doc: " + retornoDOC + "\n\n")
            return retu_dict

        elif self.classificar == 3:  #chain de salvar progresso
            if len(Memoria) == 0:
                contexto_classi_questao =  f"\n \n aluno: {input}"
            else:  contexto_classi_questao = Memoria[-1]["assistant"] + f"\n \n aluno: {input}" 
           
            Resposta_salva_prog = saveProgress.run_conversation(contexto_classi_questao)
            Memoria.append({"user": input, "assistant": Resposta_salva_prog })
            retu_dict = {"user": input, "assistant": Resposta_salva_prog }
            return  retu_dict

        else:
             # chain de conversa normal
             template = materia + template0
             prompt = PromptTemplate(
             input_variables=["chat_history", "human_input"], 
             template=template0)
             llm_chain0 = LLMChain(
             llm=ChatOpenAI(model = "gpt-3.5-turbo", temperature=0), 
             prompt=prompt, 
             verbose=False, 
             
             )
             messages = ["user: {} \n assistant: {}\n".format(dic['user'], dic['assistant']) for dic in Memoria]
             output = "".join(messages)
            # print("contexto parsed"+ output)
             with get_openai_callback() as cb:
                resposta0 = llm_chain0({"human_input" : input, "chat_history": output })
                print(type(resposta0)) # dictionary, pode remover o  json load
                total_cost +=cb.total_cost
                total_tokens +=cb.total_tokens
                prompt_tokens +=cb.prompt_tokens
                completion_tokens +=cb.completion_tokens
                text = resposta0['text']
                Memoria.append({"user": input, "assistant": text})
                retu_dict =  {"user": input, "assistant": text}
                return retu_dict


"""def chat_to_json_api(memoria):
    current_dict = memoria[len(memoria)-1]
    url = "teste"
    json_output = json.dump(current_dict)
    response = requests.post(url, json_output)

    if response.status_code == 201:
        print('Successfully created resource.')     
        print('Response JSON:', response.json())
        return True
    else:
        print('Failed to create resource.')
        print('Status Code:', response.status_code)
        print('Response JSON:', response.json())
        return False"""

def chat_bot(User_input: str):
    chatInstance = chatBot(User_input)
    Bot_Answer = chatInstance.chatModel()
    return Bot_Answer  #returns a dictionary


             
inputUser = ""

import gradio as gr
import time

theme = gr.themes.Default(primary_hue="zinc", secondary_hue="amber").set(  
    input_background_fill="#FFFFFF",
    background_fill_primary ="#abdbe3",
    button_primary_background_fill ="#e28743",
    button_primary_border_color_hover="#e28743",
)

"""block =gr.Blocks(theme=theme)

with block:   #gera block do gradio para o front end 

    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=650)
    msg = gr.Textbox()
    #clear = gr.ClearButton([msg, chatbot])
    submit= gr.Button("Submit")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        inputUser = history[-1][0]
        #print("user input"+ inputUser)
        teste = chatBot(inputUser)
        resposta_bot= teste.chatModel()  #resposta bot é o dictionary com a perguntar do 'user' e a resposta do 'assisten'
        print(resposta_bot)
        resposta_bot = resposta_bot['assistant']
        mem_dict =  Memoria[len(Memoria)-1]
        #print(mem_dict.keys())
        history[-1][1] = ""
        for character in resposta_bot:
            history[-1][1] += character
            time.sleep(0.003)
            yield history

    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    
"""
    
#block.queue()
#block.launch(inline=True, height=1000)

        
print(f"Total Tokens: {total_tokens}")
print(f"Prompt Tokens: {prompt_tokens}")
print(f"Completion Tokens: {completion_tokens}")
print(f"Total Cost (USD): ${total_cost}")


   