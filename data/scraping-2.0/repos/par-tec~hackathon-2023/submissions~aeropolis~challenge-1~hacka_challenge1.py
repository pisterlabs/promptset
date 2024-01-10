import os
import openai


def ask_ai(prompt: str, chat_content: str = "Sei un assistente AI, che aiuta la gente a generare idee",  temp: float = 0.7, stop_word: str = "\n\n", my_engine: str = "GPT"):
    """
    execute LLM interaction using a prompt and applying a role to the AI assistant.

    The prompt contains the following elements:
    - request: the request to the AI (e.g., "Tell me what is the meaning of life, based to the below considerations:")
    - text: the text to analyze (e.g., "Life is a wonderful not one-way journey. It is a journey that we can enjoy only if we are able to understand the meaning of life.")
    """    
    # set openai configuration
    openai.api_type = "azure"
    
    openai.api_base = "https://saopenai.openai.azure.com/"    
    openai.api_version = "2023-07-01-preview"
    
    #set api key from environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    eng = "saGPT"
    mod = None

    if my_engine == "LLAMA":        
        # LLAMA
        openai.api_type = "open_ai"
        openai.api_base = "http://51.159.159.233:3001/v1" 
        openai.api_version = ""
        eng = None
        mod = "/models/llama-2-13b-chat.bin"

    response = openai.ChatCompletion.create(
        engine = eng, 
        model  = mod,
        messages=[
            {
                "role": "system",
                "content": chat_content,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temp,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=stop_word,        
    )

    res = response["choices"][0]["message"]["content"]  
    
    return res



def generate_ideas(eng: str = "GPT", style = "Le idee generate devono evocare emozioni positive ed essere divertenti"):
    #Define AI role
    chatRole = "Sei un assistente AI, che aiuta la gente a generare idee"

    # Define the text of the short story that you want to use as the source to generate ideas 
    text = f"""Genera delle idee in breve frasi che userò per scrivere un racconto fantastico che parla del Mastro di Chiavi, custode di Aeropolis e possessore delle chiavi per aprire la cupola di vetro che protegge Aeropolis, la città sospesa tra le nuovole. Il suo grande amico è il Cuoco che cerca di rubargli le chiavi ma non ci riesce mai. {style}
    Esempi di idee:
    1.  Il Mastro di Chiavi è un uomo misterioso, con un'aura sinistra che fa tremare le ossa a chiunque gli si avvicini troppo.
    2.  Il Mastro di Chiavi custodisce un segreto antico che potrebbe cambiare il destino di Aeropolis per sempre. Il Cuoco vorrebbe tanto scoprire questo segreto.
    
    Idee suggerite:
    1.

    """
        
    ideas = ask_ai(text)
    print(f"--------------idee:\n {ideas}")

    return


generate_ideas("GPT", style = "Le idee generate devono evocare emozioni positive ed essere divertenti")

