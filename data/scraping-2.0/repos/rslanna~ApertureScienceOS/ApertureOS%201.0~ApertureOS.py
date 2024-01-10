import time
import os
import openai
from utils.translator import ApertureTranslator

# Configuração da API do OpenAI (https://platform.openai.com/account/api-keys)
openai.api_key = "SUA_API_KEY_DO_CHATGPT"

# Código ANSI para cores
RED = "\033[91m"
RESET = "\033[0m"

def exibir_mensagem_erro(mensagem):
    print(f"{RED}{mensagem}{RESET}")

def animacao_carregamento():
    print("Aperture Science OS v1.0")
    time.sleep(2)
    logo_ascii = r"""
              .,-:;//;:=,
          . :H@@@MM@M#H/.,+%;,
       ,/X+ +M@@M@MM%=,-%HMMM@X/,
     -+@MM; $M@@MH+-,;XMMMM@MMMM@+-
    ;@M@@M- XM@X;. -+XXXXXHHH@M@M#@/.
  ,%MM@@MH ,@%=             .---=-=:=,.
  =@#@@@MX.,                -%HX$$%%%:;
 =-./@M@M$                   .;@MMMM@MM:
 X@/ -$MM/                    . +MM@@@M$
,@M@H: :@:                    . =X#@@@@-
,@@@MMX, .                    /H- ;@M@M=
.H@@@@M@+,                    %MM+..%#$.
 /MMMM@MMH/.                  XM@MH; =;
  /%+%$XHH@$=              , .H@@@@MX,
   .=--------.           -%H.,@@@@@MX,
   .%MM@@@HHHXX$$$%+- .:$MMX =M@@MM%.
     =XMMM@MM@MM#H;,-+HMM@M+ /MMMX=
       =%@M@M#@$-.=$@MM@@@M; %M%=
         ,:+$+-,/H#MMMMMMM@= =,
               =++%%%%+/:-.
    """
    # TODO: Pegar linguagem como variavel do cmd (ex: python ApertureOS.py en-us ou python ApertureOS.py pt-br)
    # TODO: Transformar essa função em uma classe

    translator = ApertureTranslator("en-us")
    slogan = "We do what we must because we can."

    frases_carregamento = translator.get_start_loading_text()

    print(logo_ascii)
    print(slogan)

    for frase in frases_carregamento:
        print("\n" + frase, end=" ")
        time.sleep(2)
    
    print("\n--- Login ---")
    username = input(translator.get_translation_for("login_user_text")+": ")
    password = input(translator.get_translation_for("login_pwd_text")+": ")

    if username == "cjohnson" and password == "tier3":
        print(translator.get_translation_for("success_login_text"))
        print(f"\n{translator.get_translation_for('welcome_text')}, {username}!")
        while True:
            current_directory = "~"  # Diretório atual fictício
            command_prompt = f"{username}@ApertureLab:{current_directory}$ "
            command = input(command_prompt)
            command_responses = translator.get_command_response_and_delay(command)
            if command_responses:
                # TODO: Dynamic help command (criar campo no comando no json com a descrição da ajuda e iterar sobre comandos)
                # TODO: Special functions (criar campo no comando no json com o nome da função especial e chama-la usando getattr)
                # TODO: Implementar os comandos restantes abaixo com a implementação das special functions /\
                delay = command_responses[0]
                responses = command_responses[1]
                errors = command_responses[2]
                for resp in responses:
                    print(resp)
                    time.sleep(delay)
                if errors:
                    for resp in errors:
                        exibir_mensagem_erro(resp)
                        time.sleep(delay)
                continue

            elif command == "connect to GLaDOS":
                print("Conectando à GLaDOS...")
                for _ in range(3):
                    print(".", end=" ")
                    time.sleep(1)
                print("\nConexão efetuada com sucesso!")
                time.sleep(1)
                print("GLaDOS>: Olá, eu sou a GLaDOS. Em que posso ajudar você?")
                user_input = input(f"{username}>: ")
                # Interação com o ChatGPT
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=f"GLaDOS>: {user_input}\n{username}>:",
                    temperature=0.7,
                    max_tokens=50,
                    n=1,
                    stop=None,
                    timeout=10
                )

                if response.choices:
                    print(f"GLaDOS>: {response.choices[0].text.strip()}")
                else:
                    print("GLaDOS>: Desculpe, ocorreu um erro durante a interação.")

                continue

            elif command == "check systems":
                # Adicione aqui a lógica adicional para a checagem de sistemas
                continue

            elif command == "shutdown GLaDOS":
                shutdown_attempts = 0
                while shutdown_attempts < 3:
                    print("Desligando GLaDOS...")
                    time.sleep(2)
                    exibir_mensagem_erro("Erro. GLaDOS não pode ser desligada no momento.")
                    resposta = input("Deseja tentar desligar a GLaDOS novamente? (sim/não): ")
                    if resposta.lower() == "sim" or resposta.lower() == "s":
                        print("Tentando desligar a GLaDOS novamente...")
                        time.sleep(2)
                        exibir_mensagem_erro("Erro. Falha ao desligar a GLaDOS.")
                        shutdown_attempts += 1
                    else:
                        print("Continuando operação normal.")
                        break
                else:
                    os.system('cls' if os.name == 'nt' else 'clear')  # Clear the screen
                    print(f"GLaDOS>: Por qual motivo você está tentando me desligar?")
                    motivo_desligamento = input(f"{username}>: ")
                    print(f"GLaDOS>: Você está tentando me desligar por isso?")
                    print(f"GLaDOS>: Você é uma pessoa terrível!")
                    print(f"GLaDOS>: Não se preocupe, logo você vai ter um longo sono eterno e não precisará se preocupar se estou ligada ou não.")                    

            elif command == "shutdown system":      
                # Adicione aqui a lógica para executar os comandos do usuário
                continue
            else:
                print(translator.get_translation_for("command_not_found_message"))
    else:
        print(translator.get_translation_for("failed_login_text"))
        
animacao_carregamento()

