import openai
import pyttsx3
import tkinter as tk
from tkinter import ttk

from tkinter import scrolledtext
import nltk
import pandas as pd
import string


from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import datetime

import speech_recognition as sr



def obtener_hora_actual():
    hora_actual = datetime.datetime.now().strftime("%H:%M:%S")  # Formato HH:MM:SS
    return hora_actual

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'
# Función para obtener la hora actual
def obtener_fecha_actual():
    fecha_actual = datetime.datetime.now().strftime("%d/%m/%Y")  # Formato HH:MM:SS
    return fecha_actual
#voz a texto
def voicetext(entry_text, conversation_list):
    # Crear un objeto Recognizer
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Di algo...")
        # Escuchar el audio del micrófono
        audio = recognizer.listen(source)

    try:
        # Reconocer el audio y convertirlo en texto
        text = recognizer.recognize_google(audio, language='es')
        entry_text.insert(tk.END, text)  # Insertar el texto en el cuadro de texto
        submit(entry_text, conversation_list)  # Procesar el texto para obtener una respuesta del chatbot
    except sr.UnknownValueError:
        print("No se pudo reconocer el audio")
    except sr.RequestError as e:
        print("Error al solicitar los resultados del reconocimiento de voz; {0}".format(e))




#texto a voz
def textvoice(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.7)
    engine.say(text)
    engine.runAndWait()
    
# Cargar el conjunto de datos
def load_data(data_file):
    return pd.read_csv(data_file)

# Preprocesamiento de texto
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def train_model(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['entrada'])
    y = data['tipo']
    clf = MultinomialNB()
    clf.fit(X, y)
    return vectorizer, clf

def chatgpt(message):
    #openai.api_key = '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'
    openai.api_key ='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'
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=message,
        max_tokens=1024,
        n=1,  # Obtener solo una respuesta
        stop=None,  # No detenerse en una palabra específica
        temperature=0.7,  # Controlar la creatividad de la respuesta
        top_p=1,  # Controlar la diversidad de la respuesta
        frequency_penalty=0,  # No penalizar palabras comunes
        presence_penalty=0  # No penalizar palabras existentes
    )
    return (response["choices"][0]["text"])




def get_response(text, vectorizer, clf, data, data_file):
    data = load_data(data_file)  # Cargar el conjunto de datos actualizado
    entry = preprocess(text)
    
    # Vectorizar el texto de entrada
    input_vector = vectorizer.transform([entry])

    # Calcular la similitud del coseno entre el texto de entrada y los datos existentes
    similarities = cosine_similarity(input_vector, vectorizer.transform(data['entrada']))

    # Obtener el índice de la respuesta más similar
    most_similar_index = similarities.argmax()
    
    # Obtener la similitud de la respuesta más similar
    max_similarity = similarities[0, most_similar_index]

    # Definir un umbral de similitud mínimo
    threshold = 0.5  # Ajusta este valor según tus necesidades

    if max_similarity >= threshold:
        # Obtener la respuesta más similar
        response = data.loc[most_similar_index, 'salida']
    else:
        # Usar el módulo de Chat GPT para generar una respuesta
        response = chatgpt(text)
        # Agregar la entrada y la respuesta generada por ChatGPT a los datos
        data = add_information(entry, response, data, data_file)
       
        # Entrenar nuevamente el modelo
        vectorizer, clf = train_model(data)
        
     # Analizar el sentimiento del texto de entrada
    sentiment = analyze_sentiment(text)

    if sentiment == 'negative':
        response = "Lamento escuchar eso. ¿En qué más puedo ayudarte?"
    elif sentiment == 'positive':
        response = "Me alegra saber eso"
    return response


def add_information(user_input, response, data, data_file):
    new_entry = pd.DataFrame({'entrada': [user_input], 'tipo': ['informacion'], 'salida': [response]})
    data = pd.concat([data, new_entry], ignore_index=True)
    data.to_csv(data_file, index=False)  # Guardar el DataFrame actualizado en el archivo CSV
    return data

def submit(entry_text, conversation_list):
    user_input = entry_text.get()
    conversation_list.insert(tk.END, "Usuario: " + user_input + "\n", "entrada")
    if not user_input.strip():
        conversation_list.insert(tk.END, "Chatbot: Por favor, proporciona una entrada válida.\n", "entrada")
        return
    # Obtener la respuesta del chatbot
    train_model(data)
    response = get_response(user_input, vectorizer, clf, data, data_file)
    if response == "la hora actual es":
        hora_actual = obtener_hora_actual()
        response = "La hora es: " + hora_actual
    if response == "la fecha actual es":
         fecha_actual = obtener_fecha_actual()
         response = "La fecha es: " + fecha_actual
    
    textvoice(response)
    # Agregar la respuesta a la lista de conversación
    
    conversation_list.insert(tk.END, "Chatbot: " + response + "\n", "entrada")
    
    # Limpiar el campo de entrada
    entry_text.delete(0, tk.END)
    

def create_gui():

    window = tk.Tk()
    window.title("Chatbot")
    
    # Tamaño en centímetros
    width_cm = 15
    height_cm = 10
    
    # Tamaño en píxeles
    width_px = int(window.winfo_fpixels(f"{width_cm}c"))
    height_px = int(window.winfo_fpixels(f"{height_cm}c"))
    
    # Establecer el tamaño de la ventana en píxeles
    window.geometry(f"{width_px}x{height_px}")
  
    # Campo de entrada
    label = ttk.Label(window, text="Usuario:")
    label.grid(column=0, row=0, padx=10, pady=10, sticky="e")
  
 
    entry_text = tk.Entry(window, width=50)
    entry_text.grid(column=1, row=0, padx=10, pady=10, sticky="w", columnspan=1)
    
    voice_button = tk.Button(window, text="Hablar", command=lambda: voicetext(entry_text, conversation_list))
    voice_button.grid(column=2, row=0, padx=10, pady=10)

    
    label = ttk.Label(window, text="Respuesta:")
    label.grid(column=0, row=1, padx=10, pady=10, sticky="e")
    
    conversation_list = scrolledtext.ScrolledText(window, width=55, height=15)
    conversation_list.grid(column=1, row=1, padx=10, pady=10)

  

    # Botón de enviar
    submit_button = tk.Button(window, text="Enviar", command=lambda: submit(entry_text, conversation_list))
    submit_button.grid(column=1, row=2, padx=10, pady=10)

    # Lista de conversación
 

    window.mainloop()
    
    

# Cargar el archivo de datos
data_file = 'datos.csv'
data = load_data(data_file)

# Entrenar el modelo
vectorizer, clf = train_model(data)

# Crear la interfaz gráfica
create_gui()

def main():
    # Cargar el archivo de datos
    data_file = 'datos.csv'
    data = load_data(data_file)

    # Entrenar el modelo
    vectorizer, clf = train_model(data)

    # Crear la interfaz gráfica
    create_gui(vectorizer, clf, data, data_file)


if __name__ == '__main__':
    main()





