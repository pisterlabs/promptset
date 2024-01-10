import os
import json

from dotenv import load_dotenv
import openai
import datetime

load_dotenv()

# Set up OpenAI configuration
openai.api_key = os.getenv("KEY")

async def get_chat_completion(messages):
    response = await openai.Completion.create(
        engine="text-davinci-003",
        messages=messages,
    )
    return response.choices[0].message

async def main():
    history = []
    x = 0
    while x < 1:
        user_input = """Você deve retornar o que pedi na ordem que pedi e APENAS OS NUMEROS. Caso não encontre o que pedi retorne "NOT". Retorne os resultados em formato de 
        Objeto Javascript, onde as propriedades do objeto devem ser exatamente: {
            nota: "numero encontrado",
            emissao: "data encontrada",
            valorServico: "valor encontrado",
            valorImpostos: "valor encontrado,
            discriminacao: "discriminações encontradas"
        }.

        Seja sucinta e objetiva, e localize apenas as informações Número da nota, data de emissão, valor dos serviços, valor dos impostos e discriminação dos serviços no texto a seguir:   NFS-e - NOTA FISCAL DE SERVIÇOS ELETRÔNICA - RPS 1660 Série 1, emitido em: 30/03/2022   NAMAH LTDA   R TOMAZ GONZAGA, 530 APT 1800   LOURDES - Belo Horizonte - MG - 30180143   TELEFONE: 3125149565   EMAIL: CONTATO@ACENDASUALUZ.COM.BR   CNPJ: 44.713.203/0001-09   INSCRIÇÃO MUNICIPAL: 13579130012   NÚMERO DA NOTA   202200000001659   COMPETÊNCIA   30/03/2022 00:00:00   CÓDIGO DE VERIFICAÇÃO   9c9091c4   DATA DE EMISSÃO   30/03/2022 21:42:44   DADOS DO TOMADOR   NOME / RAZÃO SOCIAL   Vânia Moreira   E-MAIL   vm.vania.moreira@gmail.com   TELEFONE   16997330060   ENDEREÇO   Não informado, 01   BAIRRO / DISTRITO   Não informado   CEP   00000000   MUNICÍPIO   Belo Horizonte   UF   MG   PAÍS   Brasil   CPF / CNPJ / OUTROS   115.171.148-96   INSCRIÇÃO MUNICIPAL INSCRIÇÃO ESTADUAL   DISCRIMINAÇÃO DOS SERVIÇOS   Workshop de Autoestima.   CÓDIGO DO SERVIÇO   8.02 / 080200188 - INSTRUÇÃO E TREINAMENTO, AVALIAÇÃO DE CONHECIMENTOS DE QUAISQUER NATUREZA   MUNICÍPIO ONDE O SERVIÇO FOI PRESTADO   3106200 / Belo Horizonte   NATUREZA DA OPERAÇÃO   Tributação no municipio   REGIME ESPECIAL DE TRIBUTAÇÃO: -   VALOR DOS SERVIÇOS: 350,00   (-) DESCONTOS: 0 (-) DEDUÇÕES: 0   (-) RETENÇÕES FEDERAIS: 0,00 (=) BASE DE CÁLCULO: 350,00   (-) ISS RETIDO NA FONTE: 0 (x) ALÍQUOTA: 3,00 %   VALOR LÍQUIDO: 350,00 (=) VALOR DO ISS: R$ 10,50   RETENÇÕES FEDERAIS   PIS: R$ 0,00 COFINS: R$ 0,00 IR: R$ 0,00 CSLL: R$ 0,00 INSS: R$ 0,00   OUTRAS INFORMAÇÕES   Valor aprox dos tributos: R$ 0,00 federal, R$ 0,00 estadual e R$ 0,00 municipal Fonte: IBPT 02C353.   powered by eNotas Gateway  Retorne o resultado em formato de planilha, com as informações em colunas."""

        messages = []
        for input_text, completion_text in history:
            messages.append({"role": "system", "content": input_text})
            messages.append({"role": "user", "content": completion_text})

        messages.append({"role": "user", "content": user_input})

        try:
            response = await get_chat_completion(messages)

            completion_text = response["content"]
            print(completion_text)

            x += 1
            history.append((user_input, completion_text))

            # Parse the completion text and extract the required information
            nota = "NOT"
            emissao = "NOT"
            valorServico = "NOT"
            valorImpostos = "NOT"
            discriminacao = "NOT"

            # Note: You would need to write the code here to extract the required information from the completion text
            # and assign it to the respective variables (nota, emissao, valorServico, valorImpostos, discriminacao)

            # Save the returned object to a .py file
            objetoRetorno = {
                "nota": nota,
                "emissao": emissao,
                "valorServico": valorServico,
                "valorImpostos": valorImpostos,
                "discriminacao": discriminacao
            }

            filename = f"retorno-{objetoRetorno['nota']}.py"
            with open(filename, "w") as file:
                file.write(f"objetoRetorno = {json.dumps(objetoRetorno, indent=4)}")

            print(f"Objeto retornado salvo em {filename}")

        except Exception as error:
            print(error)

asyncio.run(main())
