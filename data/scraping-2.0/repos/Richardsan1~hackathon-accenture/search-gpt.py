'''
Artigos:

Accelerating Innovation With Generative AI: AI-Augmented Digital Prototyping and Innovation Methods - https://ieeexplore.ieee.org/document/10115412
An Enhanced AI-Based Network Intrusion Detection System Using Generative Adversarial Networks - https://ieeexplore.ieee.org/document/9908159
Automatic Chinese Meme Generation Using Deep Neural Networks - https://ieeexplore.ieee.org/document/9611242
A Survey on ChatGPT: AI-Generated Contents, Challenges, and Solutions - https://ieeexplore.ieee.org/document/10221755
A Systematic Literature Review on Text Generation Using Deep Neural Network Models - https://ieeexplore.ieee.org/document/9771452
Can Machines Tell Stories? A Comparative Study of Deep Neural Language Models and Metrics - https://ieeexplore.ieee.org/document/9194709
Chat2VIS: Generating Data Visualizations via Natural Language Using ChatGPT, Codex and GPT-3 Large Language Models - https://ieeexplore.ieee.org/document/10121440
Data Curation and Quality Evaluation for Machine Learning-Based Cyber Intrusion Detection - https://ieeexplore.ieee.org/document/9907008
Denoising-Based Decoupling-Contrastive Learning for Ubiquitous Synthetic Face Images - https://ieeexplore.ieee.org/document/10262007
Evaluating Differentially Private Generative Adversarial Networks Over Membership Inference Attack - https://ieeexplore.ieee.org/document/9656913
From ChatGPT to ThreatGPT: Impact of Generative AI in Cybersecurity and Privacy - https://ieeexplore.ieee.org/document/10198233
Generating Role-Playing Game Quests With GPT Language Models - https://ieeexplore.ieee.org/document/9980408
Hierarchical Reinforcement Learning With Guidance for Multi-Domain Dialogue Policy - https://ieeexplore.ieee.org/document/10011569
IRWoZ: Constructing an Industrial Robot Wizard-of-OZ Dialoguing Dataset - https://ieeexplore.ieee.org/document/10076424
K-LM: Knowledge Augmenting in Language Models Within the Scholarly Domain - https://ieeexplore.ieee.org/document/9866735
Leveraging Symbolic Knowledge Bases for Commonsense Natural Language Inference using Pattern Theory - https://ieeexplore.ieee.org/document/10158053
Membership Inference Attacks With Token-Level Deduplication on Korean Language Models - https://ieeexplore.ieee.org/document/10025743
Static Malware Detection Using Stacked BiLSTM and GPT-2 - https://ieeexplore.ieee.org/document/9785789
TaughtNet: Learning Multi-Task Biomedical Named Entity Recognition From Single-Task Teachers - https://ieeexplore.ieee.org/document/10041925
ToD4IR: A Humanised Task-Oriented Dialogue System for Industrial Robots - https://ieeexplore.ieee.org/document/9869659


Total de paginas lidas: 323 paginas

Custo total: 0.20 R$


'''
#Bibliotecas utilizadas
import os
from dotenv import load_dotenv
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader

load_dotenv()

chave = os.getenv("OPENAI_API_KEY")
openai.api_key = chave

#Função para usar a API da OpenAi
def Perguntar(prompt,persona):
	completion = openai.ChatCompletion.create(
		model="gpt-4",
		stop=None,
		messages=[{"role":"system", "content": persona},
		{"role": "user", "content": prompt}]
	)
	return completion['choices'][0]['message']['content']


path = os.getcwd()
#print(path)
print("Começou a ler o PDF.(Processo leva até 2 minutos para documento teste)")
#Lê o arquivo
loader = PyPDFLoader("articles/Gabriel Tardochi Salles....pdf")
#Organiza em pedaços com o numero da página
pages = loader.load_and_split()
print("Criando o VectorStore. (Processo leva até 40 segundos para documento teste)")
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings(model="text-embedding-ada-002"))
print("Terminou de ler o pdf.\nFazendo a Requisição a API.")

pergunta="O que é o chat gpt?"
print(f"fazendo a primeira pergunta modelo:\n{pergunta} \n\n")
#Loop de perguntas
while True:
	bibliografia =""
	termo_pesquisa =""
	#procurando os 3 termos com maior proximidade com a pergunta.
	docs = faiss_index.similarity_search(pergunta, k=3)
	#Pega o conteudo guardado
	for doc in docs:
		termo_pesquisa+= "Conteúdo: " + str(doc.page_content)+ "\n\n"
		bibliografia+= f"Pagina: {doc.metadata['page']} \nConteúdo:\n{doc.page_content} \n\n"
	#Prepara a pergunta
	termo_pesquisa+= f"Use este conteúdo acima se ele tiver relevancia para pergunta abaixo:\n{pergunta}"
	#Melhora a personalidade
	persona = "Você é um especialista dem Generative Ai. Você deve responderde maneira clara, pensando bem antes de escrever."
	#Envia a requisição
	final = Perguntar(termo_pesquisa,persona)
	resposta = final
	#Resposta
	print(resposta+"\n\n")
	print("Referência:\n"+bibliografia)
	pergunta = input("Qual a sua pergunta: ")
