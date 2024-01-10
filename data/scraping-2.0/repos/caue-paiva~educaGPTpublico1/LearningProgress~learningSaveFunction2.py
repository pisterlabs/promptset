import openai
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

UserINput="""(Enem/2016) No aniversário do primeiro decênio da Marcha sobre Roma, em outubro de 1932, Mussolini irá inaugurar sua Via dell Impero; a nova Via Sacra do Fascismo, ornada com estátuas de César, Augusto, Trajano, servirá ao culto do antigo e à glória do Império Romano e de espaço comemorativo do ufanismo italiano. Às sombras do passado recriado ergue-se a nova Roma, que pode vangloriar-se e celebrar seus imperadores e homens fortes; seus grandes poetas e apólogos como Horácio e Virgílio.

SILVA, G. História antiga e usos do passado um estudo de apropriações da Antiguidade sob o regime de Vichy. São Paulo: Annablume, 2007 (adaptado).

A retomada da Antiguidade clássica pela perspectiva do patrimônio cultural foi realizada com o objetivo de

a) afirmar o ideário cristão para reconquistar a grandeza perdida.
b) utilizar os vestígios restaurados para justificar o regime político.
c) difundir os saberes ancestrais para moralizar os costumes sociais.
d) refazer o urbanismo clássico para favorecer a participação política.
e) recompor a organização republicana para fortalecer a administração estatal.


aluno: a resposta é c)
"""

#primeiro numero é o total de questões e o segundo o número de acertos;
Progresso_materias = {
    "Historia_Brasil": [0, 0],
    "Historia_antiga": [0, 0],
    "Idade_media": [0, 0]
}

Sub_Topicos_Globais =["Historia_Brasil","Historia_antiga","Idade_media"]

def Salva_progresso(Subtopico, resposta_correta, Resposta_aluno):
  print(resposta_correta)
  print(Resposta_aluno)
  print(Subtopico)
  "Identifica o subTopico da questão e se o aluno acertou a questão ou não"
  Progresso_materias[Subtopico][0] += 1
  if (resposta_correta == Resposta_aluno):
   Progresso_materias[Subtopico][1] += 1
   print("acertou")

  return f"progresso salvo no subtopico {Subtopico}"

def Mostra_progressoSub(Subtopico):
  totalQuest = Progresso_materias[Subtopico][0]
  Acertos = Progresso_materias[Subtopico][1]
  porce_acertos = ((Acertos/totalQuest)*100)
  resposta = f"No Subtopico {Subtopico}, você fez {totalQuest} questões e teve {porce_acertos}% de acertos "
  print(resposta)

def Mostra_progressoGeral():
  TotalQuest = 0
  TotalAcertos = 0 
  for key in Progresso_materias:
   
    TotalQuest += Progresso_materias[key][0]
    TotalAcertos += Progresso_materias[key][1]
    
  porce_acertos = ((TotalAcertos/TotalQuest)*100)
  resposta = f"No Total,  você fez {TotalQuest} questões e teve {porce_acertos}% de acertos "
  return resposta 

#Pense passo a passo e decida se a resposta do aluno corresponde ou não à resposta verdadeira da questão, caso a resposta verdadeira não esteja disponível 
  
def run_conversation (Userinput):
 response = openai.ChatCompletion.create(
  model = "gpt-3.5-turbo-0613",
  messages = [{"role":"assistant", "content": """pense passo a passo, leia o que é pedido non enunciado e responda a alternativa correta questão, essa resposta será parte da sua resposta final  """ },
    {"role":"user", "content": Userinput }
              
              
              
              
              
              ],
  functions =[ 
     {
        "name": "Salva_progresso",
        "description": "Analiza o tópico de uma questão  e diz qual o tópico da questão e se o aluno escolheu a alternativa correta da questão",
        "parameters": {
            "type": "object",
            "properties": {
                "Subtopico": {
                    "type": "string",
                    "enum": ["Historia_Brasil","Historia_antiga","Idade_media"],
                    "description": "identifique qual o subtopico/tema da questão entre as opções "
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
 #print(message) 

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
    print(Mostra_progressoGeral())
    return Parsed_response

#print(run_conversation())

#print("\n \n \n \n")
historiaTotal = Progresso_materias["Historia_Brasil"][0]
historiaCertas = Progresso_materias["Historia_Brasil"][1]

#print(historiaTotal)
#print(historiaCertas)

#print(Mostra_progressoGeral())