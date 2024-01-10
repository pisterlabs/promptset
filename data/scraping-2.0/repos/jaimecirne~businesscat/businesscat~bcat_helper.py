import sys
import os
import logging
import pandas as pd
from openpyxl import Workbook
import ocrspace
from bcat_pagamento_dao import (obter_pagamentos)
from langchain.chains import create_extraction_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI


# enable logging
logging.basicConfig(
    # filename=f"log {__name__} pix2txt_bot.log",
    format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# get logger
logger = logging.getLogger(__name__)

api = ocrspace.API()

def respond_error(error_details):
    logger.error(f"{error_details}", exc_info=True)
    sys.exit(1)


def convrt_return_txt(pic_location):
    logger.info(f"extracting text from picture")
    try:
        output = api.ocr_file(open(pic_location, 'rb'))
    except Exception as e:
        logger.error(f"unable to extract text from picture -- {e}", exc_info=True)
        raise ValueError("Não to conseguindo entender esse print, eu sou só um gato")
    else:
        if output:
            logger.info(f"successfully extracted text from picture, returning result to bot")
            return output
        else:
            logger.error(f"The OUTPUT is None -- {output}, something definitely went wrong", exc_info=True)
            raise ValueError("Não to conseguindo entender esse print, eu sou só um gato")
    finally:
        clean_up_pix(pic_location)


def gerar_planilha():
    pagamentos = obter_pagamentos()

    if not pagamentos:
        raise ValueError("Não há dados para gerar a planilha.")
        
    # Criar um DataFrame com os dados
    colunas = ["ID", "User ID", "Nome", "Valor", "Estabelecimento", "Categoria", "Quando"]
    df = pd.DataFrame(pagamentos, columns=colunas)

    # Criar um arquivo Excel
    arquivo_excel = "planilha_pagamentos.xlsx"
    writer = pd.ExcelWriter(arquivo_excel, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Pagamentos')
    writer.close()
    
    return arquivo_excel


def clean_up_pix(picture_file):
    logger.info("attempting to delete picture file from server")
    try:
        os.remove(picture_file)
    except Exception as e:
        logger.error(f"unable to delete pic file -- {e}", exc_info=True)
    else:
        logger.info(f"successfully delete pic file {picture_file}")
    finally:
        return
