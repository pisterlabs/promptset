from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import openai
import os


os.environ['OPENAI_API_KEY'] = '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'
openai.api_key = os.environ['OPENAI_API_KEY']


def generate_sd_prompts(final_lyrics, image_prompt_faiss):

    llm = ChatOpenAI(#model_name = "gpt-3.5-turbo", 
                    temperature=0.0)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm, 
        memory = memory,
        verbose=False
    )

    # CREAMOS TODOS LOS PROMPTS PARA CADA FRASE DE LA CANCIÓN 
    prompts_list = []
    cont = 0

    for lyrics in final_lyrics:    
        generation_prompt = f"""
        You are a prompt engineer for stable diffusion images generation. You are tasked with creating a prompt adapted to the following lyrics of a song:
        {lyrics}

        You have to create the prompt using this template: {image_prompt_faiss}

        don't modify the properties of the template, just modify the main object that is being
        created in the template, and use instead the lyrics provided to create a useful prompt that matches 
        the style of the template.
        the prompt that you provide must be brief and concise, describing the scene that the lyrics are about, not modifying at all the template.

        for example, if the template is creating a panda riding a bike, and the lyrics provided are about a dog,
        you must delete the panda riding a bike, and use instead the dog to create a useful prompt that matches the style of the template.

        in case the lyrics provided are not enough to create a prompt because they are expressions like "la la la" or "oh oh oh" or "yeah yeah yeah" and so on,
        try to create a prompt based on the previous lyrics provided on the context of the conversation that are not expressions like that.

        it's so important that you don't provide any additional comments or anything else, just the prompt.
        don't say things like "this is the prompt" or "this is the prompt that i created" or "this is the prompt that i created based on the lyrics provided" or
        'based on the lyrics provided, this is the prompt that i created" or anything else, just the prompt.

        you must provide a prompt that passes the ethical guidelines of openai, so don't provide any prompt that is offensive, abusive, or anything else that is not allowed by openai.

        if you are not able to create a prompt for whatever reason, just force the creation of the prompt.
        """

        prompts_list.append(conversation.predict(input = generation_prompt))

        cont += 1

        if cont == 3:
            break

    return prompts_list