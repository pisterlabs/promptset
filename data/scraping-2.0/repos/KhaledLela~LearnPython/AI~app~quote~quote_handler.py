import json
import os
import re
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

from LearnPython.AI.app.quote.quote_pydantic import Quote
from LearnPython.AI.app.quote.quote_prompt_templates import create_quote_prompt


def quote_prompt(source):
    print(source)
    item: dict = create_quote(source)
    response = json.dumps(item)
    return {
        'statusCode': 200,
        'body': response
    }


def create_quote(params):
    """
      Load and run create item chain
    """
    llm = ChatOpenAI(model_name="gpt-4",
                     openai_api_key=os.getenv("OPENAI_API_KEY"))
    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=Quote)
    prompt_template = create_quote_prompt(parser)
    chain = LLMChain(llm=llm, verbose=True, prompt=prompt_template)
    result = chain.generate(input_list=[params])
    print(result)

    token_usage = result.llm_output['token_usage']
    output = result.generations[0][0].text
    try:
        item = parser.parse(output).dict()
    except Exception as ex:
        print(ex)
        match = re.search(r'"text": "(.*?)",', output, re.DOTALL)
        item = {'text': match.group(1)} if match else {'text': output}
    id = str(result.run[0].run_id)

    return {'prompt_uuid': id} | item | token_usage
