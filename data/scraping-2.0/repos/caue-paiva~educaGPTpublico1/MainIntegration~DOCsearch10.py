from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
load_dotenv("/home/kap/Desktop/pythonGPT/keys.env")



memoria_chat_DOC =[]  #lista que guarda a memória do chat, dictionary de 3 keys: contexto do doc, input do user e resposta da IA
loader = TextLoader(file_path="MainIntegration/enemTeste1.txt")
documents = loader.load()
llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature=0) # tentar mudar temperatura para testes.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1800, #cada chunk de texto tem no máximo 1600 caracteres, melhor numero que eu testei ate agora
    #mudar entre linux e windows parece resultar em problemas no textspitter, no windows o melhor parece ser 1600 caracteres, e no linux 1800 caracteres
    chunk_overlap  = 0,
    separators=["(Enem/"]  #divide partes do texto ao encontrar essa string, agrupa pedaços de texto entre a occorrencia dessa string
    
)

String_respostas = "(RESPOSTA)  alternativa sempre correta: "

texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)




from langchain.chains import LLMChain
prompt_template = """ você é um tutor que vai ajudar o aluno nos estudos sobre as questões ou topicos abaixo, pense passo a passo e explique o contexto geral da questão, auxiliando o aluno, VOCE NUNCA VAI DAR A RESPOSTA DA QUESTAO

O aluno não quer que voce crie novas questões a unica questão abordada será dada abaixo, não crie novas questões, nem copie essa questão nem suas alternativas,  apenas explique de forma correta ela para o aluno:

aqui está a questão {context}

aluno: {topic} 
Fale sobre o assunto tema gera abordado pela questão acima para o aluno sem dar detalhes relacionados à resposta, deixe o aluno pensar na resposta por si só, apenas o ajude a entender o contexto geral da questão, NUNCA DE A RESPOSTA DA QUESTÃO.
tutor: 
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "topic"]
)

chain = LLMChain(llm=llm, prompt=PROMPT, verbose=True)

def Pesquisa_Questoes(topic):
  
     docs = docsearch.similarity_search(topic, k=1)  #gera 1 pagina de resultado na busca por documentos
     inputs = [{"context": doc.page_content, "topic": topic} for doc in docs] #extrai um dictionary dos resultados da busca nos docs
     Quest_Docs = inputs[0]['context'] 
        #extrai o texto em si do resultado
        
     if not("(Enem/") in Quest_Docs:
      Quest_Docs  =  " \n (Enem/" + Quest_Docs  #adicionar (Enem/ caso o input não tenha isso  

     
     memoria_chat_DOC.append({"context":Quest_Docs, "UserInput": topic})     #adiciona o resultado na lista de memória
     Quest_forma_memo=""        
     for doc in memoria_chat_DOC:        #formatar os conteúdos da lista de memória para dar como contexto pro LLM
        Quest_forma_memo += (doc["context"] + "\n")
        Quest_forma_memo += (doc["UserInput"] + "\n")
     i=1
     while ("(RE" in Quest_Docs):
       Quest_Docs = Quest_Docs[:-i] 
       i+=1
    


     memoria_chat_DOC[-1]["context"] += (" \n \n"+ "resposta do tutor :  " )  
     retornoFinal = "\nQuestão do ENEM: "+ Quest_Docs 

     return memoria_chat_DOC, retornoFinal
    
