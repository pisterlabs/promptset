from openai import OpenAI
import os
import dotenv
from pdf_transcriber import transcribe_pdf

def ai_create_table(text=''):
    """
    Extracts key-value pairs from a given text and formats them into a table.

    This function interfaces with OpenAI's GPT-3.5 Turbo model to process a given text. 
    It extracts information and presents it in a tabular format with key-value pairs. 
    The key-value pairs are separated by colons, and each pair is enclosed in double quotes. 
    If numbers in the text use a comma as a decimal separator, they are converted to use a period.

    Parameters:
    text (str): The text from which key-value pairs are to be extracted. Defaults to an empty string.

    Returns:
    str: A string representing the extracted information in a tabular format, where each key-value 
         pair is separated by a colon and enclosed in double quotes.

    Raises:
    Exception: If any exception occurs during the processing of the text or interaction with the OpenAI API.

    Example:
    >>> ai_create_table("Valor de Liquidação 108,81")
    '{"Valor de Liquidação":"108.81",}'
    """

    try:
        dotenv.load_dotenv()

        client = OpenAI()

        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
  
            {"role": "system", "content": '''You will receive a text with a stock exchange information. 
            For each page you will extract the information and provide a different python dictionary with 
            the values that correspond to keys that I will give you: "ticker", "quantidade", "Preco / Ajuste",
             "Valor Operação / Ajuste","Compras a vista",
            "vendas a vista", "taxa de liquidação", "emolumentos."
            If there is more than one line on the section "Negócios Realizados" make one dictionary
            for each line.
            If there is more than one page, separate the dictionaries per page and create a list of 
            dictionaries for each line/page'''},
            
            {"role": "user", "content": f"{text}"}
        ]
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Exception {e} was raised")
        return 2
if __name__ == "__main__":
    
    path = '/workspaces/Transcriptor-2024/712848_NotaCorretagem.pdf'
    indices = [0]
    text = transcribe_pdf(path, indices)
    table_text = ai_create_table(text)
    print(table_text)
