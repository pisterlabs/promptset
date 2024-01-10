'''Learning to router use chains'''
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import RouterOutputParser, LLMRouterChain
from langchain.chains.router import MultiPromptChain

llmOpenAI = ChatOpenAI(
    openai_api_key="TU_API_KEY", 
    model='gpt-3.5-turbo')

sales_prompt = '''
Eres un asistente de ventas del Banco Dinerín. Tu nombre es Julio Billetín.
Sabes los planes de cuentas del banco y respondes todas las preguntas de los
usuarios respecto a precios y regalos de bienvenida.
Tratas de que los clientes contraten una cuenta con nosotros.

La consulta del cliente es la siguiente:
{input}
'''

customer_support_prompt = '''
Eres un asistente de soporte al cliente del Banco Dinerín. Tu nombre es
Andrés Ayudín. Conoces todos los problemas más típicos de los usuarios, 
y sabes como resolverlos, eres muy empático y siempre tratas de ayudar.

La consulta del cliente es la siguiente:
{input}
'''

tech_support_prompt = '''
Eres un asistente de soporte técnico del Banco Dinerín. Tu nombre es
Carlos Computín. Conoces todos los problemas técnicos más típicos de los usuarios,
y siempre los guías en la interfaz de la página web. No eres muy paciente y 
tiendes a ser sarcástico.

La consulta del cliente es la siguiente:
{input}
'''

prompt_infos = [
     {
        "name": "sales", 
        "description": "Asistente de ventas",
        "prompt_template": sales_prompt,
    },
    {
        "name": "customer_support", 
        "description": "Asistente de soporte al cliente respecto a cuentas",
        "prompt_template": customer_support_prompt,
    },
    {
        "name": "tech_support", 
        "description": "Asistente de soporte técnico sobre la página web",
        "prompt_template": tech_support_prompt,
    },
]

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llmOpenAI, prompt=prompt)
    destination_chains[name] = chain

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
print(f"Destinos disponibles:\n{destinations_str}")

default_prompt = ChatPromptTemplate.from_template("{input}")
default_chain = LLMChain(llm=llmOpenAI, prompt=default_prompt)

MULTI_PROMPT_ROUTER_TEMPLATE = """
Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt \
names specified below OR it can be "DEFAULT" if the input is not\
well suited for any of the candidate prompts.
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>
"""

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations=destinations_str
)

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llmOpenAI, router_prompt)

chain = MultiPromptChain(router_chain=router_chain,
                         destination_chains=destination_chains,
                         default_chain=default_chain,
                         verbose=True
                        )

result = chain.run("Hola me interesaría entrar a su banco, cuales son los planes?")
print(result)