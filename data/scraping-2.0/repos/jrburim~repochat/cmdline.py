import os
import pickle

# Importando as bibliotecas necessárias
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from deeplake.core.dataset import Dataset
import inquirer
import shutil

# Importando as funções definidas em outro arquivo
from functions import custo_embeddings_repo, db_add_repo_files, download_and_extract_repo

# Verificando se a chave da API da OpenAI está definida
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_API_KEY is None:
    print("Chave da API não encontrada")
    exit(-1)

# Definindo as embeddings que serão usadas (neste caso, as embeddings da OpenAI)
EMBEDDINGS = OpenAIEmbeddings(disallowed_special=())

print("Inicializando banco de dados...")

# Removendo o arquivo de bloqueio do banco de dados, se existir
arquivo_lock = "deeplake/dataset_lock.lock"
if os.path.exists(arquivo_lock):
    os.remove(arquivo_lock)

# Inicializando o banco de dados
db = DeepLake(embedding_function=EMBEDDINGS)

print("Banco de dados inicializado! ")

# Definindo o arquivo onde a lista de repositórios será armazenada
REPO_LIST_FILE = "repos_list.pkl"

# Inicializando a lista de repositórios
repos_list = [ ]
repoName = None
qa_chain = None

# Carregando a lista de repositórios do arquivo, se ele existir
if os.path.exists(REPO_LIST_FILE):
    with open(REPO_LIST_FILE, 'rb') as file:
        repos_list = pickle.load(file)

# Função para selecionar um repositório
def seleciona_repo():
    global repoName
    global repos_list
    global qa_chain    

    # Pergunta ao usuário qual repositório ele deseja trabalhar
    questions = [
        inquirer.List('repo',
                    message="Qual repositório você deseja trabalhar?",
                    choices=repos_list + ["Outro..."],
                ),
    ]

    answers = inquirer.prompt(questions)
    repoName = answers['repo']

    # Se o usuário escolher "Outro...", pede para ele digitar a URL do repositório
    if repoName == "Outro...":
        questions = [
            inquirer.Text('repoURL',
                        message="Digite a URL do repositório"),
        ]

        answers_other = inquirer.prompt(questions)

        # Adiciona o repositório ao banco de dados
        repoURL = answers_other['repoURL']

        assert repoURL is not None, "URL do repositório vazia, abortando..."
        repoName, destination_folder = download_and_extract_repo(repoURL)

        # Calcula o custo de adicionar o repositório ao banco de dados
        total_tokens, custoUSD = custo_embeddings_repo(destination_folder)

        # Exibe o custo em USD
        print(f"Número total de tokens: {total_tokens}")
        print(f"Custo em USD: {custoUSD:.2f}")

        # Confirma a geração dos embeddings
        questions = [
            inquirer.Confirm('confirmacao',
                            message="Deseja gerar os embeddings?",
                            default=True),
        ]

        answers_other = inquirer.prompt(questions)
        confirmacao = answers_other['confirmacao']

        if confirmacao:
            # Gera os embeddings
            db_add_repo_files(db, repoName, destination_folder)
            
            # Adicionando na lista de repositórios e salvando no disco
            repos_list.append(repoName)
            with open(REPO_LIST_FILE, 'wb') as file:
                pickle.dump(repos_list, file)

            # Apagando a pasta do repositório 
            shutil.rmtree(destination_folder)    

            print("Embeddings gerados com sucesso!")
        else:
            print("Geração dos embeddings cancelada.")
            exit(0)

    # Seleciona o repositório escolhido pelo usuário
    retriever = db.as_retriever( search_kwargs={'distance_metric': 'cos', 'fetch_k': 100, 'maximal_marginal_relevance': True, 'k': 10, 'filter': {'metadata': { 'repo': repoName }}})
    model = ChatOpenAI(model='gpt-3.5-turbo')
    qa_chain = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

# Chamando a função para selecionar um repositório
seleciona_repo()

# Inicializando o histórico do chat
chat_history = []

print("Inicializando chatbot...")
# Loop principal do chatbot
while True:
    print(" para sair, digite \"exit\" ou \"sair\", para voltar a seleção de repositório, digite \"voltar\"")
    question = input(f"({repoName}) - Digite sua pergunta: ")
    if question == "exit"  or question == "sair" or question == "":
        break;
    if question == "voltar":
        seleciona_repo()
        chat_history = []
        continue
    result = qa_chain({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))    
    print(f" >>>>> : {result['answer']} \n")