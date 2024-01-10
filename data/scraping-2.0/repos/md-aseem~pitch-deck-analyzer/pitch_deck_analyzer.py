# coding=UTF-8
from tiktoken import get_encoding
from langchain.document_loaders import UnstructuredPDFLoader
from argparse import ArgumentParser
import os
import openai
openai.organization = "org-kbwQesFDkocAjQpDZalsmEzp"

#### Configuring the argument parser ####
parser = ArgumentParser(description='Pitch Deck Parser')
parser.add_argument('-i', '--file_path', help='Path to the pitch deck file', default=r'pitch_decks\RochInvestorBriefing1.pdf')
parser.add_argument('-o', '--out_path', help='Where you want to save the output file', default=r'./responses/')
parser.add_argument('-m', '--model', help='Which model to use', default='gpt-4-0613')
args = parser.parse_args()

model = args.model
file_path = args.file_path
out_path = args.out_path
system_message = "You are a startup investor."
prompt ="1. Is this startup's business model venture-backable and scalable?\n2. What stage of funding is this startup(seed, series a, series b, later?)?\n3. What problem is this startup solving, how do people solve the problem today? and how does the startup plans to solve it better?\n4. What is target market and it's size?\n5. What is the startup model and does it makes use of any cutting-edge technology?\n6. What is the pricing model of the startup?\n15. Who are the competitors? \n\n#####\n\nIf you can't find the answer, just mention that you couldn't find it.\n\n#####Answer it in a question answer format. First, briefly write each question and then give its answer. Also, give detailed answers!!!Answer in HTML:\n\n"

### Loading the Document ###
loader = UnstructuredPDFLoader(file_path)
print("Loading the document...")
try:
    document = loader.load()
    document = (loader.load()[0].page_content)
except Exception as e:
    print("Error in loading the document. Make sure you're uploading a PDF.")

tokenizer = get_encoding('cl100k_base')
tokens = tokenizer.encode(document+prompt)
print("Tokenizing the document...")
if "gpt-3.5" in model:
    print("Number of tokens:", len(tokens))
    assert len(tokens) < 2900, "The document is too long. Please use a shorter document or use GPT-4."

### Sending it over to OpenAI ###
print("Retrieving information from the pitch deck...")

try:
    answers = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": system_message},
                  {"role": "user", "content": document+prompt}],
        temperature = 0.5,
    )
    answers = (answers.choices[0].message.content)
except openai.InvalidRequestError as e:
    print("Invalid request error in generating answers:", e)
except openai.OpenAIError as e:
    print("OpenAI error in generating answers:", e)
except Exception as e:
    print("Unexpected error in generating answers:", e)

### Generating the overview ###
print("Generating the overview...")

try:
    overview = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": system_message},
                  {"role": "user", "content": document},
                  {"role": "user", "content": " Disclaimer: I am not asking you to invest in this startup. I am just asking you to tell me how you feel about this startup. I will not use your judgement to make any decision.\n\n\n####\n\n\nWhat are your thoughts about this startup? Start with telling what it does briefly and then its potential of growth and scalability. Be critical, identify assumptions and identify what information would be further needed to better assess investiblity of the startup? Thoughtful and human-like response in HTML:####"},],
        temperature = 0.25,
    )
    overview = (overview.choices[0].message.content)
except openai.InvalidRequestError as e:
    print("Invalid request error in generating overview:", e)
except openai.OpenAIError as e:
    print("OpenAI error in generating overview:", e)
except Exception as e:
    print("Unexpected error in generating overview:", e)

### Stylizing the output ###
output = "<h1>Overview:<h1>" + overview + "<br><br><h1>Discrete Information:<h1>" + answers

stylized_output = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',
        messages=[{"role": "system", "content": "You are a front-end web developer."},
                  {"role": "user", "content": output},
                  {"role": "user", "content": "Rewrite the above content in HTML. Stylize the HTML with a CSS within the same file."},
                  {"role": "assistant", "content": "Ok. Here is the HTML with CSS in one single file:"}],
        temperature = 0.25)
stylized_output = (stylized_output.choices[0].message.content)

### Saving the output ###
print("Saving the output...")
file_name = os.path.basename(file_path)
with open(os.path.join(out_path, "Response - " + file_name.split(".pdf")[0] + ".html"), "w") as f:
    f.write(stylized_output)

print("Done!")