import openai
from halo import Halo
from promptss import getRecipePrompt
import os
import PyPDF2
import re 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

openai.api_key = config.OPENAI_API_KEY
spinner = Halo(text='Loading', spinner='dots')

def chat_gpt_completion(chat_message, append = False, usr_prompt = ""):
    
    spinner.start("text generating")
        
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages= chat_message
    )
    
    spinner.succeed("text generated")
    
    response = completion['choices'][0]["message"]["content"]
    
    if append:
        chat_message.append({"role": "system", "content": response})
        
    
    return response, chat_message


def recipe_maker(food, output_name, cuisine, protien, made_recipes, message):
    if message == []:
        initPrompt = getRecipePrompt(food,cuisine, protien, made_recipes)
        message = [{"role": "user", "content": initPrompt}]
    else: 
        message.append({"role": "user", "content": "Another recipe please"})
        
    r, m = chat_gpt_completion(message, append=True)

    def convert_markdown_pdf (input_file, output_file):
    
        os.system(f'mdpdf -o {output_file} {input_file}')
        
    with open(f"{output_name}.md", "w") as file:
        # Write a string to the file
        file.write(r)
        
    convert_markdown_pdf(f"{output_name}.md", f"{output_name}.pdf")
    
    return m 
    
    
def merge_pdfs(pdf_files, output_file):
    pdf_merger = PyPDF2.PdfMerger()

    for pdf in pdf_files:
        pdf_merger.append(pdf)

    pdf_merger.write(output_file)
    pdf_merger.close()
    
    
def extract_heading(markdown_file):
    heading_pattern = re.compile(r"^#\s+(.+)", re.MULTILINE)

    with open(markdown_file, "r") as md_file:
        content = md_file.read()

    match = heading_pattern.search(content)

    if match:
        return match.group(1)
    else:
        return None

    
def cookbook(numRecipes, name, cuisine = False, protien = False):
    files = []
    filesb = []
    message = []
    listOfMadeRecipes = ""
    for x in range(numRecipes):
        files += [f"recipe{x}.pdf"]
        filesb += [f"recipe{x}.md"]
        
        message = recipe_maker(name, f"recipe{x}", cuisine, protien, listOfMadeRecipes, message)
        
    merge_pdfs(files, "files/cookbook.pdf")
    delete_files(files + filesb)
    


def delete_files(files_to_delete):
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except FileNotFoundError:
            print(f"File not found: {file}")
        except PermissionError:
            print(f"Permission denied: {file}")
        except OSError as e:
            print(f"Error deleting file {file}: {e}")