import tika
from tika import parser
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inicializar o parser Tika
tika.initVM()

# Extrair o conteúdo do PDF
file_data = parser.from_file('livro.pdf')
pdf_content = file_data['content']

# Pre-processamento de texto
# Quebrar o conteúdo do PDF em frases
sentences = sent_tokenize(pdf_content)

# Tokenização das palavras em cada frase
words = []
for sent in sentences:
    words.extend(word_tokenize(sent.lower()))

# Remover stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

# Stemming (reduzir a palavra ao seu radical)
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]

# Criar uma lista de frases limpas
clean_sentences = [' '.join(stemmed_words[i:i+100]) for i in range(0, len(stemmed_words), 100)]

# Vetorizar o conteúdo do PDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(clean_sentences)

# Função para responder perguntas
def respond(question):
    # Pré-processamento da pergunta
    question_words = word_tokenize(question.lower())
    filtered_question_words = [word for word in question_words if word not in stop_words]
    stemmed_question_words = [stemmer.stem(word) for word in filtered_question_words]
    clean_question = ' '.join(stemmed_question_words)

    # Transformar a pergunta em um vetor TF-IDF
    question_vector = vectorizer.transform([clean_question])

    # Calcular a similaridade entre a pergunta e as frases do PDF
    similarities = cosine_similarity(question_vector, tfidf_matrix)

    # Obter a frase mais similar ao vetor de pergunta
    most_similar_sentence = clean_sentences[similarities.argmax()]

    return most_similar_sentence

# Exemplo de uso do chatbot
while True:
    user_input = input("Faça uma pergunta: ")
    if user_input.lower() == "sair":
        break
    response = respond(user_input)
    print(response)