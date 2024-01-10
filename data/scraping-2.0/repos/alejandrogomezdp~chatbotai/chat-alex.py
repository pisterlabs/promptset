from flask import Flask, request, render_template, jsonify
import openai
import re

app = Flask(__name__)

openai.api_key = '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'

def format_response(response):
    # Detecta y formatea bloques de código
    response = re.sub(r'```(.*?)```', r'<pre><code>\1</code></pre>', response, flags=re.DOTALL)

    # Divide el texto en párrafos
    paragraphs = response.split('\n\n')

    # Combina los párrafos formateados en una sola cadena
    formatted_response = '<p>'.join(paragraphs)

    return formatted_response

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.json['message']
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto en programación reconocido y tutor de codificación. Estás ayudando a un profesional con un problema de codificación."},
                {"role": "user", "content": user_input}
            ]
        )
        response = response.choices[0].message['content']
        response = format_response(response)  # Formatea la respuesta
        return jsonify({'response': response})

    return render_template('chat.html')

@app.route('/registro', methods=['GET', 'POST'])
def registro():
    # Aquí va tu código para el registro de usuarios
    pass

@app.route('/iniciar-sesion', methods=['GET', 'POST'])
def iniciar_sesion():
    # Aquí va tu código para iniciar sesión
    pass

if __name__ == '__main__':
    app.run(debug=True)
