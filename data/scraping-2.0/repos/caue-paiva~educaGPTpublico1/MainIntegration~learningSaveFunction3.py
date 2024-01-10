import openai
import json
import os
from dotenv import load_dotenv
load_dotenv("/home/kap/Desktop/pythonGPT/keys.env")


openai.api_key = os.getenv("OPENAI_API_KEY")


# no json, na lista de cada disciplina, o primeiro numero é o total de questões e o segundo o número de acertos;

Sub_Topicos_Globais =["Historia_Brasil","Historia_antiga","Idade_media" , "Geografia_Fisica", "Geografia_Politi/social" ]

def Salva_progresso(Subtopico, resposta_correta, Resposta_aluno):
  print("função de progresso iniciada")
  Resposta_aluno = Resposta_aluno.lower()
  with open('MainIntegration/json_userData.json', 'r') as fp:
   dados = json.load(fp)
   "Identifica o subTopico da questão e se o aluno acertou a questão ou não"
   dados[Subtopico][0] += 1
   dados["total_questoes"] += 1
   if (resposta_correta == Resposta_aluno):
    dados[Subtopico][1] += 1
    dados["total_acertos"] += 1
    print("acertou")

    
  with open('MainIntegration/json_userData.json', 'w') as fp:
        json.dump(dados, fp) 
  return f"progresso salvo no subtopico {Subtopico}"

# def Mostra_progressoSub(Subtopico):
#   totalQuest = Progresso_materias[Subtopico][0]
#   Acertos = Progresso_materias[Subtopico][1]
#   porce_acertos = ((Acertos/totalQuest)*100)
#   resposta = f"No Subtopico {Subtopico}, você fez {totalQuest} questões e teve {porce_acertos}% de acertos "
#   print(resposta)

def Mostra_progressoGeral():
  TotalQuest = 0
  TotalAcertos = 0 
  with open("MainIntegration/json_userData.json", 'r') as fp:
   dados = json.load(fp)
   TotalQuest = dados["total_questoes"]
   TotalAcertos = dados["total_acertos"]
  
   
  porce_acertos = ((TotalAcertos/TotalQuest)*100)
  resposta = f"No Total,  você fez {TotalQuest} questões e teve {porce_acertos:.2f}% de acertos "
  return resposta 

#Pense passo a passo e decida se a resposta do aluno corresponde ou não à resposta verdadeira da questão, caso a resposta verdadeira não esteja disponível 
  
def run_conversation (Userinput):
 print("função de conversa iniciada")
 response = openai.ChatCompletion.create(
  model = "gpt-3.5-turbo-0613",
  messages = [{"role":"assistant", "content": """pense passo a passo, leia o que é pedido non enunciado e responda a alternativa correta questão, essa resposta será parte da sua resposta final  """ },
    {"role":"user", "content": Userinput }
              
              
              
              
              
              ],
  functions =[ 
     {
        "name": "Salva_progresso",
        "description": "Analiza o tópico de uma questão  e diz qual o tópico da questão e se o aluno escolheu a alternativa correta da questão, lembre se que não importa se a resposta for uma letra maiuscula ou minuscula:  a = A, b= B, c=C, d=D, e= E",
        "parameters": {
            "type": "object",
            "properties": {
                "Subtopico": {
                    "type": "string",
                    "enum": ["Historia_Brasil","Historia_antiga","Idade_media","Geografia_Politi/social","Geografia_Fisica"],
                    "description": "identifique qual o subtopico/tema da questão entre as opções , descreva EXATAMENTE umas das opções"
                },
                "resposta_correta": {
                    "type": "string",
                    "enum": ["a", "b", "c", "d", "e"],
                    "description": "Pense passo-a-passo, se atente na pergunta sendo feita, resolva a questão e ache a alternativa correta e retorne qual a letra correta dessa alternativa",
                },
                "Resposta_aluno":{
                    "type": "string",
                    "enum": ["a", "b", "c", "d", "e"],
                    "description": "Abaixo da questão atual o aluno vai dar a resposta dele sobre a questão, diga qual a alternativa que ele trouxe, não deixe a resposta do aluno interferir em achar a resposta certa",
            },
            },
            "required":["Subtopico", "resposta_correta","Resposta_aluno" ],
        },
    },
    
   ],
function_call="auto"

 )

 message = response["choices"][0]["message"]
 print(message) 

 if message.get("function_call"):
    function_name = message["function_call"]["name"]
    arguments = json.loads(message["function_call"]["arguments"])
    Subtopico = arguments["Subtopico"]
    Resposta_aluno = arguments["Resposta_aluno"]
    resposta_correta = arguments["resposta_correta"]
    
    #print(type(Subtopico))
    #print("teste sub" + str(Subtopico))
    function_response= Salva_progresso(
        Subtopico=str(Subtopico),
        resposta_correta=str(resposta_correta),
        Resposta_aluno=str(Resposta_aluno)

    )

    AI_response = openai.ChatCompletion.create(
      model = "gpt-3.5-turbo-0613",
      messages = [ {"role":"assistant", "content": "chame a função e depois de retornar fale se o aluno acertou ou não a questão , se ele errar mostre a resposta certa e sua lógica" },
        
        {"role":"user", "content": Userinput },
                  message, {
                     
                     "role": "function",
                     "name": function_name,
                     "content": function_response
                    },
                  
              ],
      )
    Parsed_response = AI_response["choices"][0]["message"]["content"]
    progresso_geral= Mostra_progressoGeral()
    return (progresso_geral+ "\n \n" + Parsed_response)
