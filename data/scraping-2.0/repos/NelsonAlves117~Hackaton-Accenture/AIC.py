import json

# Importa a biblioteca 'openai'.
import openai

# Importa a classe 'PdfReader' da biblioteca 'PyPDF2'.
from PyPDF2 import PdfReader

# Esta função recebe três parâmetros: o caminho do documento PDF, a página inicial e a página final.

def get_pdf_text(document_path, start_page=1, final_page=1):
  # Cria um objeto PdfReader para ler o documento PDF especificado.
  reader = PdfReader(document_path)

  # Calcula o número total de páginas no documento PDF.
  number_of_pages = len(reader.pages)

  # Extrai o texto da primeira página e armazena na variável 'page'.
  page = reader.pages[1].extract_text()

  # Itera pelas páginas especificadas, começando da página 'start_page' até a 'final_page'.
  # O loop concatena o texto de todas as páginas dentro desse intervalo.
  for page_num in range(start_page - 1, min(number_of_pages, final_page)):
    page += reader.pages[page_num].extract_text()

  # Retorna o texto extraído do PDF.
  return page

# Chama a função 'get_pdf_text' para extrair o texto de um PDF específico.
# Os resultados são armazenados nas variáveis 'page' e 'ics'.
Curriculo_Professores = get_pdf_text('PDF/Professores.pdf',1,16)
Antigas_ICs = get_pdf_text('PDF/IC.pdf',1,12)

with open("KEY/openai.json") as json_file:
    data = json.load(json_file)

openai_apikey = data['openai']['api_key']

openai.api_key = openai_apikey


# Define a função 'Assistente_IC' que recebe um parâmetro 'usuario1'.
def Assistente_IC(usuario1):

  # Define uma variável 'personalidade' que descreve o papel do assistente.
  personalidade = "Você trabalha em uma universidade e seu papel é dar 3 sugestões de temas de Iniciações Científicas a partir da área de interesse do aluno,"

  # Define uma variável 'professores' que especifica a tarefa de sugerir mentores e fornecer seus emails.
  professores = "sugerir pelo menos 3 dos melhores mentores para o tema, informe seus emails"

  # Define uma variável 'tarefas' que indica a tarefa de fornecer links de sites para inspirações sobre o tema.
  tarefas = "além disso, indique sites para pegar inspirações sobre o tema, forneça o link de cada um deles"

  # Define uma variável 'ant_ic' que instrui o assistente a verificar se os temas sugeridos já foram desenvolvidos anteriormente.
  ant_ic = "Certifique-se que suas sugestões já não tenham sido desenvolvidas anteriormente; seguem algumas Iniciações já criadas"

  # Chama a API de conclusão do OpenAI para gerar uma resposta com base nas informações fornecidas.
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=[
      # Mensagem do sistema que contém informações sobre a personalidade e tarefas do assistente.
      {"role": "system", "content": f"{personalidade}, para isso utilize as seguintes informações {Curriculo_Professores},{professores} {tarefas} caso o aluno já tenha um tema específico apenas dê os professores e os sites"},

      # Mensagem do usuário contendo a entrada 'usuario1'.
      {"role": "user", "content": usuario1}
    ]
  )

  # Retorna o conteúdo da mensagem gerada pelo assistente.
  return(completion.choices[0].message.content)

# Define a função 'IC_anteriores' que recebe um parâmetro 'usuario2'.
def IC_anteriores(usuario2):

  # Define uma variável 'personalidade' que descreve o papel do assistente.
  personalidade1 = "Você trabalha em uma universidade e seu papel é citar as Iniciações Científicas da universidade que já foram desenvolvidas em relação ao tema de interesse."

  # Define uma variável 'professores' que especifica a tarefa de mencionar orientadores e alunos das Iniciações já desenvolvidas.
  professores = "Mencione os orientadores e alunos dessas Iniciações já desenvolvidas."

  # Chama a API de conclusão do OpenAI para gerar uma resposta com base nas informações fornecidas.
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=[
      # Mensagem do sistema que contém informações sobre a personalidade e tarefas do assistente.
      {"role": "system", "content": f"{personalidade1}, para isso utilize as seguintes informações {Antigas_ICs},{professores}"},

      # Mensagem do usuário contendo a entrada 'usuario2'.
      {"role": "user", "content": usuario2}
    ]
  )

  # Retorna o conteúdo da mensagem gerada pelo assistente.
  return(completion.choices[0].message.content)

# Solicita ao usuário que forneça informações sobre seu curso e áreas de interesse e afinidade.
usuario = input("Me fale um pouco sobre seu curso e as áreas em que tenha interesse e afinidade: ")

# Chama a função 'Assistente_IC' com as informações fornecidas pelo usuário e imprime a resposta.
print(Assistente_IC(usuario))

# Chama a função 'IC_anteriores' com as informações fornecidas pelo usuário e imprime a resposta.
print(IC_anteriores(usuario))
