from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

def get_openai_model_names_list()->list:
    openai_model_names_list = ['gpt-', 'davinci', 'curie', 'babbage', 'dall-e', 'ada']
    return openai_model_names_list


def is_openai_model(model_name:str)->bool:
    openai_model_names_list = get_openai_model_names_list()

    #check if model_name contains any string openai_model_names_list
    for openai_model_name in openai_model_names_list:
        if openai_model_name in model_name:
            return True
        
    return False


def build_chat_model(opt, env):
    openai_key = env['OPENAI_API_KEY']
    model = ChatOpenAI(temperature=opt.temperature, openai_api_key=openai_key, model=opt.model_name)
    embeddings = OpenAIEmbeddings(api_key = openai_key)
    return model, embeddings

