# Flask backend

from flask import Flask, request, Response
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI

app = Flask(__name__)
llm = OpenAI(streaming=True)

@app.route('/chat', methods=['POST']) 
def chat():
  text = request.json['text']
  
  def stream(token):
    text = {'id': id, 'text': token}
    yield f'data: {json.dumps(text)}\n\n'

  id = 0
  return Response(llm.generate(text, callback=StreamingStdOutCallbackHandler(stream)),
                  mimetype='text/event-stream')

if __name__ == '__main__':
  app.run()