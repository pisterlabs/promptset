
from sqlalchemy.orm import Session
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, LLMChain
import json

dataset_path = 'hub://luisfrentzen/data'

chat_history = []

QA = None

def search_db():
    global QA

    if QA is None:
        embeddings = OpenAIEmbeddings()
        db = DeepLake(dataset_path=dataset_path, read_only=True, embedding_function=embeddings)
        retriever = db.as_retriever()
        retriever.search_kwargs['distance_metric'] = 'cos'
        retriever.search_kwargs['fetch_k'] = 100
        retriever.search_kwargs['maximal_marginal_relevance'] = True
        retriever.search_kwargs['k'] = 10
        model = OpenAI(temperature=0.0, max_tokens=-1)
        QA = ConversationalRetrievalChain.from_llm(model, retriever=retriever, return_source_documents=True)

    return QA

def question_answering(prompt: str) -> str:
    chat_history = []
    prompt = "Tuliskan semua penyakit dan masalah kesehatan dari semua dokumen yang diberikan"
    qa = search_db()
    result = qa({'question': prompt, "chat_history": chat_history})

    source_documents = result['source_documents']
    sickness = result['answer']

    print(str(result['answer']))
    print('=============================')

    template = """Tuliskan semua obat untuk mengatasi penyakit yang muncul di paragraf:  {question}"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = OpenAI(temperature=0.0, max_tokens=256)
    qaa = LLMChain(prompt=prompt, llm=llm)
    result = qaa.run(str(result['answer']))

    result = str(result)
    result = result.replace('Obat untuk ', '')
    result = result.replace('Penyakit ', '')
    result = result.replace('\n', '')
    result = result.replace('\n2', '')
    result = result.replace('\n6', '')

    drugs = result

    print(result)
    print('=============================')

    template = """given Anatomical Therapeutic Chemical (ATC) Classification System categories:

    1. M01AB - Anti-inflammatory and antirheumatic products, non-steroids, Acetic acid derivatives and related substances
    2. M01AE - Anti-inflammatory and antirheumatic products, non-steroids, Propionic acid derivatives
    3. N02BA - Other analgesics and antipyretics, Salicylic acid and derivatives
    4. N02BE/B - Other analgesics and antipyretics, Pyrazolones and Anilides
    5. N05B - Psycholeptics drugs, Anxiolytic drugs
    6. N05C - Psycholeptics drugs, Hypnotics and sedatives drugs
    7. R03 - Drugs for obstructive airway diseases
    8. R06 - Antihistamines for systemic use


    Help me categorize the drugs mentioned in this text: {question}
    
    Please answer in this format [Number]. [Disease Name]: [ATC] - [Drug Name],"""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = OpenAI(temperature=0.0, max_tokens=-1)
    qaa = LLMChain(prompt=prompt, llm=llm)
    result = qaa.run(str(result))

    sd = []
    for i in range(len(source_documents)):
        sd.append(dict(source_documents[i]))

    json_dictionary = {'source_documents': sd, 'sickness': str(sickness), 'drugs': str(drugs), 'answer': str(result)}
    json_object = json.dumps(json_dictionary, indent=4)

    with open("dataset/LLM-output.json", "w") as outfile:
        outfile.write(json_object)

    return {'message': "successfully retrieve LLM answer in LLM-output.json"}

def get_result():
    with open('dataset/LLM-output.json') as json_file:
        data = json.load(json_file)
        return data
    
def relevant_docs(keyword):
    keyword = keyword.lower()
    with open('dataset/LLM-output.json') as json_file:
        data = json.load(json_file)
        data = data['source_documents']
        sd = []
        for i in range(len(data)):
            doc = data[i]
            if doc['page_content'].lower().find(keyword) >= 0:
                sd.append(dict(doc))

    return {'source_documents': sd}
