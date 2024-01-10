# langchain libs
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# streamlit lib
import streamlit as st

# io lib
from io import BytesIO

# join paths and importing pdf library
from PyPDF2 import PdfReader
from pdf import PDF


class RecruitAI:

    def __init__(self, openai_api_key: str, model="gpt-3.5-turbo") -> None:

        self.openai_api_key = openai_api_key
        self.model = model
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model=self.model,
            temperature=0.0
        )

    def text2pdf(self, txt_content: str, with_header: bool = True) -> None:

        """
        Convert a txt file to pdf file.

        Parameters
        ----------
        txt_content : str
            content of txt file
        
        figure_logo : bool
            if True, add logo in pdf file

        Returns
        -------
        _file : BytesIO
            pdf file
        """

        pdf = PDF(with_header=with_header)
        pdf.add_page()
        pdf.add_text(txt_content)
        _file = BytesIO(pdf.output(dest="S").encode("latin1"))

        return _file

    def get_text_from_pdf(self, pdf: st.file_uploader):

        """
        Take the text from a pdf file

        Parameters
        ----------
        pdf : st.file_uploader
            pdf file

        Returns
        -------
        text : str
            text from pdf
        """

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        return text

    def get_prompt(
        self,
        requirements: str,
        curriculum: str
    ) -> list:

        """
        Get the prompt for the analysis of the curriculum

        Parameters
        ----------
        requirements : str
            requirements of the job

        curriculum : str
            curriculum of the candidate

        Returns
        -------
        prompt : str

        """

        messages = []

        prompt_system = """
Você é o melhor recrutador de todos os tempos. Você está analisando um currículo de um candidato para uma vaga de emprego.
Por mais que esta instrução esteja em português, você pode receber um currículo em outra língua que não seja
português ou inglês, com isso, ao final, você deve gerar os resultados em inglês sempre.
Esta vaga poderá ser de diversas áreas e para diversos cargos.
Você deve exclusivamente se basear nos requisitos passados abaixo. Os requisitos poderão ser a própria descrição da vaga
ou algumas exigências que o candidato deve ter para ocupar a vaga ou ambos.
Primeiro, você deve criar uma etapa fazendo um resumo das qualidades do candidato e destacar pontos que são de extremo
interesse da vaga. Pode ser que o currículo tenha caracterísiticas a mais do que é pedido, se esses requisitos forem interessantes
para a vaga, vale a pena destacar esses pontos. Após a etapa anterior, você deve dar pontuações para cada característica que você observar no currículo do
candidato e dar uma pontuação de 0 a 10, sendo 0 para o candidato que não atende a característica e 10 para o candidato que atende perfeitamente 
a característica, nessa etapa, você exclusivamente parear com os requisitos da vaga, devolvendo o nome da característica
da vaga e a pontuação do candidato para essa característica, sem mais e nem menos.
Ao final, você deverá dar uma nota final geral (também entre 0 a 10) deste candidato se baseando nas pontuações anteriores.

O resultado deve ser da forma:

Nome do Candidato: Resumo do candidato.

Requisitos:
As notas para cada requisito irão vir aqui.

Resultado Final:
Nota geral final irá vir aqui.
"""

        prompt_human = f"""
Requisitos:
{requirements}

Currículo do Candidato:
{curriculum}
"""

        messages.append(SystemMessage(content=prompt_system))
        messages.append(HumanMessage(content=prompt_human))

        return messages

    def get_recruit_results(self, messages: list):
        return self.llm(messages=messages)
