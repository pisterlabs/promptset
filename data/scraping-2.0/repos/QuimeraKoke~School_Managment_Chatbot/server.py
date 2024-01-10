# Create a flask app with just one route that return a html page name index.html and another one that return a json
from langchain import OpenAI, SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
PASSWORD = "a1s2d3f4"
db = SQLDatabase.from_uri(f"mysql+pymysql://root:{PASSWORD}@127.0.0.1/asistente_icco")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/question', methods=['POST'])
def chat_api():
    question = request.json.get('question')
    llm = OpenAI(temperature=0, openai_api_key="",
                 model_name='gpt-3.5-turbo')
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
    PROMPT = """ 
        Tienes una base de datos que se llama asistente_icco, con las siguientes tablas:
        - cursos: id_cursos: llave primaria, nombre: nombre del curso.
        - materia: id_materia: llave primaria, nombre: nombre de la materia.
        - trabajos: id_trabajo: llave primaria, nombre: nombre del trabajo, descripcion: descripción del trabajo ,id_materia: llave foranea de la materia, id_cursos: llave foranea del curso, id_tipo: llave foranea del tipo de trabajo.
        - tipo_trabajo: id_tipo: llave primaria, tipo: nombre del tipo de trabajo.
        - horario: id_horario: llave primaria, dia: dia de la semana, hora_inicio: hora de inicio, hora_fin: hora de fin, id_materia: llave foranea de la materia, id_cursos: llave foranea del curso.
        Esta base de datos representa los cursos, trabajos y horarios del colegio ICCO.
        Dada una pregunta de entrada, primero crea una consulta de MySQL sintácticamente correcta para ejecutar, 
        luego observa los resultados de la consulta y genera una respuesta en lenguaje natural en español.
        La pregunta:: {question}
        """
    answer = db_chain.run(PROMPT.format(question=question))
    return jsonify({'answer': answer})


if __name__ == '__main__':
    app.run(debug=True)
