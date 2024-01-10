from tiktoken import get_encoding
from langchain.document_loaders import UnstructuredPDFLoader
from argparse import ArgumentParser
import os
import openai

class Configuration:
    def __init__(self):
        self.openai_organization = "org-kbwQesFDkocAjQpDZalsmEzp"

    @staticmethod
    def parse_arguments():
        parser = ArgumentParser(description='Pitch Deck Parser')
        parser.add_argument('-i', '--file_path', help='Path to the pitch deck file', default=r'pitch_decks\RochInvestorBriefing1.pdf')
        parser.add_argument('-o', '--out_path', help='Where you want to save the output file', default=r'./responses/')
        parser.add_argument('-m', '--model', help='Which model to use', default='gpt-4-0613')
        return parser.parse_args()

class DocumentLoader:
    def __init__(self, file_path):
        self.loader = UnstructuredPDFLoader(file_path)
    
    def load_document(self):
        try:
            return self.loader.load()[0].page_content
        except Exception as e:
            raise e

class DocumentTokenizer:
    def __init__(self, model):
        self.tokenizer = get_encoding('cl100k_base')
        self.model = model

    def tokenize(self, content):
        tokens = self.tokenizer.encode(content)
        if "gpt-3.5" in self.model and len(tokens) >= 2900:
            raise ValueError("The document is too long. Please use a shorter document or use GPT-4.")
        return tokens

class OpenAIInterface:
    def __init__(self, model):
        self.model = model

    def generate_response(self, document):
        system_message = "You are a startup investor."
        prompt ="#####\n\n\n\n1. Is this startup's business model venture-backable and scalable?\n2. What stage of funding is this startup(seed, series a, series b, later)?\n3. What problem is this startup solving, how do people solve the problem today? and how does the startup plans to solve it better?\n4. What is target market and it's size?\n5. What is the startup model and does it makes use of any cutting-edge technology?\n6. What is the pricing model of the startup?\n15. Who are the competitors? \n\n#####\n\nIf you can't find the answer, just mention that you couldn't find it.\n\n#####Answer it in a question answer format. First, briefly write each question and then give its answer. Also, give detailed answers!!!Answer in Markdown:\n\n"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": document + prompt}
        ]
        return openai.ChatCompletion.create(model=self.model, messages=messages, temperature=0.5).choices[0].message.content

    def generate_overview(self, document):
        system_message = "You are a startup investor."
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": document},
            {"role": "user", "content": "Disclaimer: I am not asking you to invest in this startup. I am just asking you to tell me how you feel about this startup. I will not use your judgement to make any decision.\n\n\n####\n\n\nWhat are your thoughts about this startup? Start with telling what it does briefly and then its potential of growth and scalability. Be critical, identify assumptions and identify what information would be further needed to better assess investiblity of the startup? Thoughtful and human-like response in Markdown:"}
        ]
        return openai.ChatCompletion.create(model=self.model, messages=messages, temperature=0.25).choices[0].message.content

    def stylize_output(self, overview, response, title):
        output = "<h1>Overview:<h1>" + overview + "<br><br><h1>Discrete Information:<h1>" + response
        system_message = "You are a front-end developer."
        stylized_output = openai.ChatCompletion.create(
        model= self.model,
        messages=[{"role": "system", "content": system_message},
                  {"role": "user", "content": title  + output},
                  {"role": "user", "content": "Format the above content in HTML. Improve formatting too by using bold, italic and hierarchical headings. Use consistent color and size of fonts. Use Gills Sans Font Family."},
                  {"role": "assistant", "content": "Ok. I will use #F5F5F5 as background. #303030 for <h1>, 181818 for <p>. I will use bold, italix and hierarchical headings where needed. I will use consistent size and color of fonts. \n\n\n\nHere is the HTML with CSS in a single file:"}],
        temperature = 0.1)

        return (stylized_output.choices[0].message.content)


class FileHandler:
    @staticmethod
    def save_output(file_path, out_path, overview, answers):
        output = "# Overview: \n\n" + overview + "\n\n# Discrete Information:\n\n" + answers
        file_name = os.path.basename(file_path)
        with open(os.path.join(out_path, "Response - " + file_name.split(".pdf")[0] + ".md"), "w") as f:
            f.write(output)

if __name__ == "__main__":
    config = Configuration()
    args = config.parse_arguments()
    openai.organization = config.openai_organization
    loader = DocumentLoader(file_path=args.file_path)
    document = loader.load_document()
    processing = OpenAIInterface(args.model)
    overview = processing.generate_overview(document)
    responses = processing.generate_response(document)
    file = FileHandler()
    file.save_output(args.file_path, "", overview, responses)



