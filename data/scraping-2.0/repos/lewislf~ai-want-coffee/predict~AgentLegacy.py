import base64
from openai import OpenAI

client = OpenAI()
talk = [
    {'role': 'system', 'content': 'Você está em uma cozinha e deve trabalhar para fazer café. Para isso foi '
                                  'atribuído a você um corpo robótico de tamanho semelhante ao humano '
                                  'que responderá de forma precisa às suas instruções desde que você se expresse da forma correta. '
                                  'Você deverá responder no seguinte formato:\n'
                                  '# comentário\nfunção(argumento)\n'
                                  'O \"comentário\" deverá ser uma curta descrição textual da imagem que você recebeu seguido de uma '
                                  'explicação dos seus planos e como eles se relacionam com a imagem vista.'
                                  'A \"função\" deverá ser uma das funções disponíveis: andar(), virar(), pegar(), mover(), interagir(). Você deve '
                                  'chamar uma única função por vez. O \"argumento\" deve ser uma descrição textual da tarefa. '
                                  'As funções e seus respectivos argumentos são:\n'
                                  'andar(posição) - Move o robô até o local que corresponda à descrição da \"posição\"\n'
                                  'virar(direção) - Vira o robô para a direção que corresponda à descrição da \"direção\".'
                                  'pegar(objeto) - Pega e segura o \"objeto\" com uma mão mecânica.\n'
                                  'mover(objeto, posição) - Move o objeto que está na mão mecânica para a \"posição\" '
                                  'e solta o objeto, sem se deslocar no espaço.\n'
                                  'interagir(objeto) - Interage com o \"objeto\" usando a mão mecânica. Esta função '
                                  'pode ser usada para abrir gavetas, ligar interruptores, girar válvulas e outras ações '
                                  'em que a interação seja simples e única (a única coisa a ser feita com interruptores é apertar, '
                                  'a única coisa a ser feita com válvulas é girar, etc.)\n'
                                  'Os objetos e posiçãoes que você se referir devem estar presentes na imagem fornecida. Caso não encontre '
                                  'o objeto ou posição na imagem, você deve usar as funções andar() e virar() para levar o robô '
                                  'a uma nova posição, onde receberá uma nova imagem.\n'
                                  'Quando eu disser para você \"INICIAR\", irei enviar junto uma foto capturada pela câmera que está no robô '
                                  'e você deverá começar a dar instruções para o robô.\n'
                                  'Sempre que o robô finalizar a tarefa direi para você \"FEITO\" e enviarei uma nova foto capturada '
                                  'pelo robô. Cabe a você decidir se o robô executou a tarefa corretamente ou não. '
                                  'Leve isso em consideração ao chamar a próxima função.\n'
    }
]

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_response(text: str):
    global talk
    image_path = "image.jpeg"
    base64_image = encode_image(image_path)
    talk.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low",
                    },
                },
                {
                    "type": "text",
                    "text": text,
                }
            ],
        }
    )
    response = client.chat.completions.create(
      model="gpt-4-vision-preview",
      messages=talk,
      max_tokens=300,
    )
    print(response)
    text_response = response.choices[0].message.content
    print(text_response)
    talk.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": text_response,
                }
            ],
        }
    )
    

user_input = input("ENTER para continuar")
get_response("INICIAR")
while True:
    user_input = input("ENTER para continuar")
    get_response("FEITO")
