import dotenv, re
from time import sleep
from typing import List
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import Document

from item import Item

dotenv.load_dotenv()

# The initialisation of the llm should be:
#    llm = OpenAI(model="davinci-instruct-beta")
# But to stay under the 3RPM we will use the sleep function
def llm (query: str) -> str :
    sleep(20)
    model = OpenAI(model="text-davinci-003")
    return model(query)

embedding = OpenAIEmbeddings()

CHROMA_DB_PERSIST_DIR = "./chroma_index"
CHROMA_DB_COLLECTION_NAME = "items"

prompt_template_select_first_item = PromptTemplate(
    template="You are a personal stylist. Your client asked for an outfit that follows this indications:\n{query}\nNow you will have to describe thoroughly how you want the outfit to be and specially you will have to describe one specific peace of the outfit. Use the followign format: \n\nOutfit Description: *description of the outfit*\nMain Peace Description: *description of the central peace of the outfit*\n\nRemember to follow your client indications and to give a descriptive response.",
    input_variables=["query"]
)

def prompt_template_generate_outfit_description (query: str, items: List[Item]) :
    item_descriptions = [ item.describe() for item in items ]

    return f"You are a personal stylist. Your client asked for an outfit that follows this indications:\n{query}\nAnd that contains this items:\n{item_descriptions}\nDescribe the following outfit and how to make it more alike your customer indications.\nOutfit Description: " 

def prompt_template_select_item(query: str, selected_items: List[Item], outfit_description:str) :
    item_descriptions = [ item.describe() for item in selected_items ]
    
    return f"You are a personal stylist. Your client asked for an outfit that follows this indications:\n{query}\nYou had the following outfit idea:\n{outfit_description}\nAnd you selected the following items for the outfit:\n{item_descriptions}\nDescribe the next clothing peace to include in the outfit. Remember to stick to your idea and your clients instructions.\n"

def prompt_template_select_item_or_end(query: str, selected_items: List[Item], outfit_description:str):
    item_descriptions = [ item.describe() for item in selected_items ]
    
    return f"You are a personal stylist. Your client asked for an outfit that follows this indications:\n{query}\nYou had the following outfit idea:\n{outfit_description}\nAnd you selected the following items for the outfit:\n{item_descriptions}\nDecide if you want to add more items to the outfit. If you do not want to, you must write End Outfit, else describe the next clothing peace for your outfit. Remember to stick to your idea and your client indications.\n"

def prompt_template_select_id_of_item(query: str, selected_items: List[Item], outfit_description: str, documents: Document, description: str) :
    item_descriptions = "\n".join([ item.describe() for item in selected_items ])
    doc_description = "\n".join([ f"Peace ID: {i + 1}\n" + Item.from_document(doc).describe() for i, doc in enumerate(documents)])

    return f"You are a personal stylist. Your client asked for an outfit that follows this indications:\n{query}.\nYou had the following outfit idea:\n{outfit_description}\nAnd you selected the following items for the outfit:\n{item_descriptions}\nYou need to select which one of the following clothing peaces matches best with the outfit and adheres to this description: {description}\nSelect one of these items:\n{doc_description}\nReason about the best match for the outfit and write its ID (when you have selected an item you must write 'Peace ID: *ID of that item*'). You can only select one item now, but keep in mind that you will be able to select them in the future.\n\nResponse:\n"

def generate(query: str, min_items: int = 2, max_items: int = 4, preselected_items: List[Item] = []) -> List[Item] :
    outfit_description = None
    selected_items = preselected_items
    chroma = Chroma(persist_directory=CHROMA_DB_PERSIST_DIR, embedding_function=embedding, collection_name=CHROMA_DB_COLLECTION_NAME)

    while len(selected_items) < max_items :
            if len(selected_items) == 0 :
                prompt = prompt_template_select_first_item.format(query=query)
                output = llm(prompt=prompt)
                
                match = re.search(r"Outfit Description:(\s*)(?P<outfit>([\s\S]*))(\s*)Main Peace Description:(\s*)(?P<main>([\s\S]*))", output)
                if match is None :
                    raise Exception(f"Invalid output format: {output}")
                
                outfit_description = match["outfit"]
                description = match["main"]
                
            elif len(selected_items) < min_items :
                if outfit_description is None :
                    prompt = prompt_template_generate_outfit_description(query, selected_items)
                    outfit_description = llm(prompt=prompt)
                prompt = prompt_template_select_item(query=query, items=selected_items, outfit_description=outfit_description)
                description = llm(prompt)
            else :
                if outfit_description is None :
                    prompt = prompt_template_generate_outfit_description(query, selected_items)
                    outfit_description = llm(prompt=prompt)
                prompt = prompt_template_select_item_or_end(query, selected_items, outfit_description)    
                description = llm(prompt) 
            
            if re.search("End Outfit", description) :
                return selected_items
            

            documents = chroma.similarity_search(query=description)

            prompt = prompt_template_select_id_of_item(query, selected_items, outfit_description, documents, description)

            output = llm(prompt=prompt)
            
            match = re.search(r"Peace ID: (\d+)", output)
            
            if match is None :
                raise Exception(f"Invalid output format: {output}")
            
            id = match[1] - 1
            
            selected_items.append(Item.from_document(document=documents[id]))
            
    return selected_items
            