import os
import shutil
from bs4 import BeautifulSoup
import requests
import io
from zipfile import ZipFile
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from deeplake.core.dataset import Dataset

EXTENSOES_DEV = ["py", "js", "ts", "html", "css", "scss", "json", "xml", "yml", "md", 
            "java", "cpp", "h", "c", "php", "rb", "go", "swift", "kt", "sql",
            "cs", "sh", "pyc", "rs", "tsx", "jsx", "sass", "less", "vue", "rbw",
            "pl", "ps1", "bat", "cmd"]

TMP_DIR = "tmp"

import requests

def main_repository_branchname(url):
    """
    Esta função é usada para detectar o nome da branch principal de um repositório do GitHub.
    Ela faz uma requisição GET para a URL do repositório e analisa o HTML da página para encontrar o nome da branch principal.

    Args:
        url (str): A URL do repositório do GitHub.

    Returns:
        str: O nome da branch principal do repositório, se encontrado. Se não for encontrado, uma exceção é lançada.
    """

    # Fazendo a requisição para obter o HTML da página
    response = requests.get(url)
    if response.status_code != 200:
        return "Erro ao acessar o repositório"

    # Analisando o HTML com BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Procurando pela tag que contém o nome do branch principal
    # O seletor usado aqui é um exemplo e pode precisar ser ajustado
    # A tag é um 'span' com a classe 'css-truncate-target' e um atributo 'data-menu-button'
    branch_tag = soup.find(lambda tag: tag.name == 'span' and tag.get('class') == ['css-truncate-target'] and tag.has_attr('data-menu-button'))

    # Se a tag foi encontrada, retorna o texto dentro dela (o nome do branch)
    # Caso contrário, lança uma exceção
    if branch_tag:
        return branch_tag.get_text(strip=True)
    else:
        raise Exception("Nome do branch principal não encontrado")


def download_and_extract_repo(url):    
    """
    Esta função é usada para fazer o download e extrair um repositório do GitHub em um diretório temporário.

    Args:
        url (str): A URL do repositório do GitHub.

    Returns:
        tuple: O nome do repositório e o caminho do diretório onde o repositório foi extraído.
    """

    # Cria um diretório temporário se não existir
    os.makedirs(TMP_DIR, exist_ok=True)

    # Extrai o nome do repositório da URL
    repo_name = url.split("/")[-1].replace(".git", "")
    
    # Obtém o nome da branch principal do repositório
    main_branch = main_repository_branchname(url)
    print(f"Branch principal: {main_branch}")

    # Faz o download do arquivo zip do repositório
    zip_url = f"{url}/archive/refs/heads/{main_branch}.zip"
    print(f"Baixando {zip_url}...")
    response = requests.get(zip_url)
    
    # Cria uma pasta com o nome do repositório e extrai o conteúdo do zip nela
    destination_folder = os.path.join(TMP_DIR, repo_name)

    # Apaga o diretório se ele já existir
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)

    os.makedirs(destination_folder, exist_ok=True)
    with ZipFile(io.BytesIO(response.content)) as zip_file:
        zip_file.extractall(destination_folder)
    
    extracted_dir = None

    # Procura o diretório que foi extraído do zip
    for dir_branch in os.listdir(destination_folder):
        if os.path.isdir(os.path.join(destination_folder, dir_branch)):
            extracted_dir = os.path.join(destination_folder, dir_branch)
            break

    if extracted_dir != None:
        # Move o conteúdo da pasta extraída para a pasta do repositório
        for dir in os.listdir(extracted_dir):
            os.rename(os.path.join(extracted_dir, dir), os.path.join(destination_folder, dir))
        # Deleta a pasta extraída
        os.rmdir(extracted_dir)

    # Retorna o nome do repositório e o caminho do diretório onde o repositório foi extraído
    return repo_name, destination_folder

def db_add_repo_files(db, repoName, repoFolder, extensoes_dev=EXTENSOES_DEV) -> Dataset:
    """
    Percorre a base de código alvo e carrega todos os arquivos
    para fragmentação e, em seguida, incorporação de texto
    """
    # Lista para armazenar os documentos
    documents = []

    # Percorre todos os arquivos no diretório do repositório
    for dirpath, dirnames, filenames in os.walk(repoFolder):
        for file in filenames:
            # Verifica se o arquivo tem uma das extensões especificadas
            if extensoes_dev is None or any(fnmatch.fnmatch(file, '*.' + ext) for ext in extensoes_dev):
                try: 
                    # Carrega o arquivo e divide em documentos
                    loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                    documents.extend(loader.load_and_split())
                except UnicodeDecodeError:
                    # Se ocorrer um erro de decodificação, tenta novamente com outro encoding
                    loader = TextLoader(os.path.join(dirpath, file), encoding='ISO-8859-1')
                    documents.extend(loader.load_and_split())    
                except Exception as e: 
                    print(e)

    # Adiciona o nome do repositório aos metadados de cada documento
    for doc in documents:
      doc.metadata['repo'] = repoName
    
    # Fragmenta os documentos
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)

    # Gera as incorporações de texto para a base de código alvo
    db.add_documents(chunks)
    return db

def get_retriever(db, repo):
    """
    Esta função cria um recuperador para um repositório específico.

    Args:
        db (Dataset): A base de dados onde os documentos estão armazenados.
        repo (str): O nome do repositório para o qual o recuperador será criado.

    Returns:
        Retriever: O recuperador criado para o repositório especificado.
    """

    # Retorna um recuperador para o repositório especificado
    # O recuperador é configurado para usar a métrica de distância cosseno, buscar os 100 documentos mais próximos,
    # usar a Relevância Marginal Máxima (Maximal Marginal Relevance - MMR) para diversificar os resultados,
    # retornar os 10 documentos mais relevantes e filtrar os documentos para incluir apenas aqueles do repositório especificado
    return db.as_retriever( search_kwargs={'distance_metric': 'cos', 'fetch_k': 100, 'maximal_marginal_relevance': True, 'k': 10, 'filter': {'metadata': { 'repo':repo }}})


def check_repo_in_db(db, repo):
    """
    Esta função verifica se um repositório específico já está na base de dados.

    Args:
        db (Dataset): A base de dados onde os documentos estão armazenados.
        repo (str): O nome do repositório a ser verificado.

    Returns:
        bool: True se o repositório estiver na base de dados, False caso contrário.
    """

    # Faz uma busca na base de dados filtrando pelos documentos do repositório especificado
    # Retorna True se algum documento for encontrado, False caso contrário
    return len(db.vectorstore.search(filter={'metadata': { 'repo':repo }})['id']) > 0

import tiktoken

def calcular_total_tokens(nome_arquivo):
    """
    Esta função calcula o total de tokens em um arquivo.

    Args:
        nome_arquivo (str): O nome do arquivo a ser analisado.

    Returns:
        int: O total de tokens no arquivo.
    """

    # Obtém o encoding do modelo cl100k_base do tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")

    # Tenta abrir o arquivo com a codificação utf-8
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
            conteudo = arquivo.read()
    # Se ocorrer um erro de decodificação, tenta abrir o arquivo com a codificação ISO-8859-1
    except UnicodeDecodeError:
        with open(nome_arquivo, 'r', encoding='ISO-8859-1') as arquivo:  # Tente uma codificação diferente
            conteudo = arquivo.read()

    # Codifica o conteúdo do arquivo em tokens
    tokens = list(encoding.encode(conteudo, disallowed_special=()))

    # Calcula o total de tokens
    total_tokens = len(tokens)

    # Retorna o total de tokens
    return total_tokens

#total = calcular_total_tokens('meu_arquivo.txt')

import os
import fnmatch

def calcular_total_tokens_diretorio(diretorio, extensoes_dev=None):   
    """
    Esta função calcula o total de tokens em todos os arquivos de um diretório.

    Args:
        diretorio (str): O diretório a ser analisado.
        extensoes_dev (list, optional): Lista de extensões de arquivo a serem consideradas. Se None, todos os arquivos serão considerados.

    Returns:
        int: O total de tokens em todos os arquivos do diretório.
    """
    total_tokens = 0
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            if extensoes_dev is None or any(fnmatch.fnmatch(file, '*.' + ext) for ext in extensoes_dev):
                total_tokens += calcular_total_tokens(os.path.join(root, file))
    return total_tokens

def custo(total_tokens):
    """
    Esta função calcula o custo em dólares para processar um determinado número de tokens.

    Args:
        total_tokens (int): O total de tokens a serem processados.

    Returns:
        float: O custo em dólares para processar o total de tokens.
    """
    #$0.0001 / 1K tokens
    return (total_tokens / 1000) * 0.0001
    
def custo_embeddings_repo(diretorio):
    """
    Esta função calcula o total de tokens e o custo em dólares para processar todos os arquivos de um diretório.

    Args:
        diretorio (str): O diretório a ser analisado.

    Returns:
        tuple: O total de tokens e o custo em dólares para processar todos os arquivos do diretório.
    """
    total_tokens = calcular_total_tokens_diretorio(diretorio, extensoes_dev=EXTENSOES_DEV)
    custoUSD = custo(total_tokens)
    return total_tokens, custoUSD


