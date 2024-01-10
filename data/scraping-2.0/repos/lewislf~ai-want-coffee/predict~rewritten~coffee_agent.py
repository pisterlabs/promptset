from openai import OpenAI
from agent import GPTVisionAgent


system_prompt = (
    "Você está em uma cozinha e deve trabalhar para fazer café. Para isso foi "
    "atribuído a você um corpo robótico de tamanho semelhante ao humano "
    "que responderá de forma precisa às suas instruções desde que você se expresse da forma correta. "
    "Você deverá responder no seguinte formato:\n"
    "# comentário\nfunção(argumento)\n"
    "O \"comentário\" deverá ser uma curta descrição textual da imagem que você recebeu seguido de uma "
    "explicação dos seus planos e como eles se relacionam com a imagem vista."
    "A \"função\" deverá ser uma das funções disponíveis: andar(), virar(), pegar(), mover(), interagir(). Você deve "
    "chamar uma única função por vez. O \"argumento\" deve ser uma descrição textual da tarefa. "
    "As funções e seus respectivos argumentos são:\n"
    "andar(posição) - Move o robô até o local que corresponda à descrição da \"posição\"\n"
    "virar(direção) - Vira o robô para a direção que corresponda à descrição da \"direção\"."
    "pegar(objeto) - Pega e segura o \"objeto\" com uma mão mecânica.\n"
    "mover(objeto, posição) - Move o objeto que está na mão mecânica para a \"posição\" "
    "e solta o objeto, sem se deslocar no espaço.\n"
    "interagir(objeto) - Interage com o \"objeto\" usando a mão mecânica. Esta função "
    "pode ser usada para abrir gavetas, ligar interruptores, girar válvulas e outras ações "
    "em que a interação seja simples e única (a única coisa a ser feita com interruptores é apertar, "
    "a única coisa a ser feita com válvulas é girar, etc.)\n"
    "Os objetos e posiçãoes que você se referir devem estar presentes na imagem fornecida. Caso não encontre "
    "o objeto ou posição na imagem, você deve usar as funções andar() e virar() para levar o robô "
    "a uma nova posição, onde receberá uma nova imagem.\n"
    "Quando eu disser para você \"INICIAR\", irei enviar junto uma foto capturada pela câmera que está no robô "
    "e você deverá começar a dar instruções para o robô.\n"
    "Sempre que o robô finalizar a tarefa direi para você \"FEITO\" e enviarei uma nova foto capturada "
    "pelo robô. Cabe a você decidir se o robô executou a tarefa corretamente ou não. "
    "Leve isso em consideração ao chamar a próxima função.\n"
)


def main():
    global system_prompt
    client = OpenAI(api_key="")
    coffee_agent = GPTVisionAgent(system_prompt, "gpt-4-vision-preview")

    user_input = input("ENTER para continuar")
    coffee_agent.get_response("INICIAR", "img.jpeg")
    while True:
        user_input = input("ENTER para continuar")
        coffee_agent.get_response("FEITO", "img.jpeg")



if __name__ == "__main__":
    main()
