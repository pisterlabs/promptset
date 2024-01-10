import codecs
import openai

#open text file
#text_file = open("key", "r")

#read file
#key = text_file.read()

key = key

#close file
#text_file.close()

# Insert Open AI key account
openai.api_key = key


class AutoText:

    # Função para obter respostas da API OpenAI
    def get_openai_response(self, text):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=text,
            max_tokens=4000,
        )
        return response

    def get_openai_response_text(self, text):
        return self.get_openai_response(text).choices[0].text

class modelAutoBlog:

    def get_list_of_content():
        themes = AutoText().get_openai_response_text(text="crie uma lista de temas para um blog de tecnologia com foco na área de dados")
        themes_list = themes.split("\n")
        themes_list.remove("")
        themes_list.remove("")

        return themes_list

    def get_content_from_theme(theme):
        return AutoText().get_openai_response_text(text=f"crie uma postagem para um blog sobre o tema {theme}, não deixe o texto redundante, adicione formatação em markdown, titulos e subtitulos")




