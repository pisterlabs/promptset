## agent telco plans

# LangChain
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.agents import load_tools
from langchain.agents import Tool
from langchain.chat_models import ChatVertexAI

#Other
from google.oauth2 import service_account
from google.cloud import aiplatform
import pandas as pd
import json
import os

with open("./keys.json", "r") as api_keys_f: 
  api_keys = json.loads(api_keys_f.read())
  for key in api_keys.keys():
    os.environ[key] = api_keys[key]

del api_keys_f, api_keys, key

search = GoogleSerperAPIWrapper()

credentials = service_account.Credentials.from_service_account_file("./gkey.json")

aiplatform.init(project="exploring-genai",
                credentials=credentials)


serper_tool = Tool(
        name="Search Answer",
        func=search.run,
        description="Útil para saber o que são os serviços de valor agregado",
)


PREFIX = """
Responda apenas com o nome do plano.
Você deve usar as ferramentas abaixo para responder à sua pergunta:

python_repl_ast: um shell Python. Use isso para executar comandos python. A entrada deve ser um comando python válido. Ao usar esta ferramenta, às vezes a saída é abreviada - certifique-se de que não pareça abreviada antes de usá-la em sua resposta.
Você está trabalhando com um dataframe do pandas em Python.
"""
FORMAT_INSTRUCTIONS = """
Use o seguinte formato:

Pergunta: a pergunta de entrada que você deve responder
Pensamento: você deve sempre pensar no que fazer
Ação: a ação a ser executada deve ser uma de [python_repl_ast]
Entrada de ação: a entrada para a ação
Observação: o resultado da ação
... (este pensamento/ação/entrada/observação de ação pode ser repetido N vezes)
Pensamento: agora sei a resposta final
Final Answare: a resposta final à pergunta de entrada original
"""
SUFFIX = """
Este é o resultado de `print(df.head())`:
{df_head}

Você está trabalhando com um dataframe do pandas em Python. O nome do dataframe é `df` e a descrição de suas colunas é a seguinte:
Nome: Relativo ao nome do plano de celular, o mesmo visto pelo usuário
Tipo: Mostra se um plano é Pós-Pago ou Pré-Pago
Categoria: Mostra se é um plano individual ou de família
Valor: Preço do plano que é pago pelo usuário
Franquia de Dados: Quantidade de dados móveis que o usuário possui mensalmente, medido em GB
Whatsapp: Mostra se um plano é possui dados ilimitados para Whatsapp ou não
Ligacoes e SMS: Mostra se um plano é possui Ligações e SMS ilimitadas ou não
Bonus Portabilidade em dados: Bonus que o usuário ganhará mensalmente na franquia de dados se vier de outra operadora
Roaming Internacional em Dados: Franquia de dados disponivel para Roaming Internacional
Perfil: Tipo de perfil de usuário, referente ao tipo de entretenimento que o usuário gosta, seprado por vírgula
Serviço de Valor Agregado: Serviços que não são da operadora mas que podem contribuir com o perfil do usuário

Começar!
Pergunta: {entrada}
{agente_scratchpad}"""

## Defining the agent
def create_agent():
        llm = ChatVertexAI(model_name="chat-bison-32k")
        llm.temperature = 0.8

        tools = load_tools(
        ["llm-math", "requests_get"],
        llm=llm
        )
        tools += [serper_tool]

        return create_pandas_dataframe_agent(llm,
                                df=pd.read_csv(".\data\PlanosCelular-Planos.csv"), 
                                agent="zero-shot-react-description",
                                extra_tools=[serper_tool],
                                handle_parsing_errors=True,
                                agent_kwargs={
                                        'prefix':PREFIX,
                                        'format_instructions':FORMAT_INSTRUCTIONS,
                                        'suffix':SUFFIX,
                                        "handle_parsing_errors":True
                                },
                                return_only_outputs=True)


#Deixar a tool mais precisa descrição de coluna, tools extras coisas assim, deixar a tabela melhor
