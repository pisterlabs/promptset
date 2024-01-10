from flask import Flask, render_template, jsonify, request, send_file
import tensorflow as tf
import io
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
from dotenv import load_dotenv
import os
from reportlab.pdfgen import canvas
from weasyprint import HTML
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})
load_dotenv()

model = tf.keras.models.load_model('model_DenseNet121_Full_Sample.h5')
loader = PyPDFLoader("disease_compendium.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)
texts = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

pinecone.init(api_key='9f6644e9-2ab1-46a5-8d35-d5ade0ee39bf', environment='gcp-starter')
index_name = pinecone.Index('lung-disease')
vectordb = Pinecone.from_documents(texts, embeddings, index_name='lung-disease')
retriever = vectordb.as_retriever()

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

def process_user_input(user_input):
    if(len(user_input)==0):
        query='''
'''
    query ='''The array provided symbolizes if the user has potentially a chest selected medically condition. The array shows 1 if the user has the corresponding disease and 0 otherwise.
    The order of diseases are No Finding, Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture, Support Devices. Based on the diseases from the array and the symptoms the user is showing, provide all the diseases and list down their symptoms, what are possible lifestyle changes and what can be the possible treatments for this.
    The order of diseases are No Finding, Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture, Support Devices. Based on the diseases from the array and the symptoms the user is showing, provide all the diseases and list down their symptoms, what are possible lifestyle changes and what can be the possible treatments for this.
    The following are some of the symptoms the user is facing: ''' + user_input
    result = chain.run({'question': query})
    return result


def generate_pdf_content(result):
    buffer = io.BytesIO()
    html_content = f"<html><body>{result}</body></html>"
    HTML(string=html_content).write_pdf(buffer)

    buffer.seek(0)
    pdf_content = buffer.read()
    base64_pdf_content = base64.b64encode(pdf_content).decode('utf-8')
    return base64_pdf_content

def generate_pdf(result, filename):
    buffer = io.BytesIO()
    html_content = f"<html><body>{result}</body></html>"
    HTML(string=html_content).write_pdf(buffer)

    buffer.seek(0)
    with open(filename, 'wb') as f:
        f.write(buffer.read())

def preprocess_image(image):
    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, [224, 224])
    image_array = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    image_array = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    return image_array
    
def predict_label(image_data):
    image = tf.image.decode_jpeg(image_data)
    preprocessed_image = preprocess_image(image)

    prediction = model.predict(tf.expand_dims(preprocessed_image, axis=0))[0]

    prediction_list = prediction.tolist()  

    return prediction_list


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image_data = file.read()

    p = predict_label(image_data)

    print("Predictions:", p)
    result = process_user_input(str(p))
    print(result)

    pdf_content = generate_pdf_content(result)
    # pdf_content = "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"
    
    return jsonify({
        'prediction': p,
        'pdfcontent': pdf_content
    })

@app.route('/output', methods=['GET'])
def output():
    pdf_filename = 'output.pdf'
    return send_file(pdf_filename, as_attachment=True)


    # The following line is removed, as it was unreachable code
    # return send_file(pdf_filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
