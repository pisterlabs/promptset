import sys
import os
import logging
from bcat_pagamento_dao import (obter_pagamentos)
from langchain.chains import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, validator, Field
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from datetime import datetime, date
from typing import Union
from dateutil import parser

from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

from typing_extensions import Annotated

def parse_data_customizada(valor: str) -> datetime:
    valor_corrigidos = valor.encode('latin-1').decode('utf-8')

    try:
        return parser.parse(valor_corrigidos)
    except ValueError:
        raise ValueError(f'Formato de data inválido: {valor_corrigidos}. Utilize um formato reconhecido.')

class Pagamento(BaseModel):
    valor: float = Field(
        title='valor',
        description='valor do pagamento feito em reais R$',
        gt=0,
    )
    quando: datetime = Field(
        title='quando',
        description='Data e hora do pagamento',
    )
    estabelecimento: str = Field(
        title='estabelecimento',
        description='Esse foi o local ou pessoa que recebeu o valor do pagamento',
    )
    categoria: str = Field(
        'outro',
        title='categoria',
        description='Essa é qualquer informação que identifique a razão do pagamento como supermercado, restaurante, combustível... caso não tenha informação usar outro',
    )

    @validator("quando", pre=True, always=True)
    def parse_data(cls, value):
        if isinstance(value, datetime):
            return value

        try:
            return parse_data_customizada(value)
        except ValueError:
            raise ValueError(f'Formato de data inválido: {value}. Utilize um formato reconhecido.')


# enable logging
logging.basicConfig(
    # filename=f"log {__name__} pix2txt_bot.log",
    format='%(asctime)s - %(funcName)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# get logger
logger = logging.getLogger(__name__)

def respond_error(error_details):
    logger.error(f"{error_details}", exc_info=True)
    sys.exit(1)


def analisar_texto_pagamento(source_text, OPENAI_API_KEY):
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(source_text)

    logger.info(texts)        

    try:

        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k", openai_api_key=OPENAI_API_KEY)

        template = """
                    Processamento Strings Pagamento
                    Você receberá strings contendo informações sobre transações de pagamento. Essas strings podem conter sujeira, erros de codificação e uma variedade de formatos de data. Sua tarefa é processar essas strings para extrair informações relevantes do pagamento, como valor, data, estabelecimento e categoria.
                    1. Certifique-se de lidar com problemas de codificação na string. Substitua ou corrija caracteres especiais, se necessário.
                    2. Extraia informações valor do pagamento, a data e hora, o estabelecimento e a categoria associada à transação.
                    3. Considere informações limitadas, como apenas o dia e o mês, e use o ano corrente e meia-noite como hora,
                    muito importante *passe a string para o formato '%Y-%m-%d %H:%M' !*
                    {input}
                    """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | llm

        text_fixed = chain.invoke({"input": texts[0]})

        chain = create_extraction_chain_pydantic(pydantic_schema=Pagamento, llm=llm)

        result = chain.run(text_fixed)

        return result[0]
    
    except Exception as e:
        logger.error(f"unable to using a openIA for analize the text from picture -- {e}", exc_info=True)
        raise ValueError("Não to conseguindo entender esse print, eu sou só um gato")
    

