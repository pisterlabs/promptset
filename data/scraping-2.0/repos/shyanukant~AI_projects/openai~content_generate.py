import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import LLMChain

def generate_content(topic, tone):
    # open ai api key
    OPENAI_KEY = os.environ.get('OPENAI_KEY')

    # crete llm model
    chat_llm = ChatOpenAI(temperature=0.7, openai_api_key=OPENAI_KEY)

    # response schema (key, value pair)
    heading_schema = ResponseSchema(name="heading", description="This is a heading")
    body_schema = ResponseSchema(name="body", description="This is a body.")

    response_schema = [ heading_schema, body_schema ]

    # generate output parser (json structure)
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schema)

    # generate json formate instruction for template string
    format_instructions = output_parser.get_format_instructions()

    # template string
    template = """You are the dedicated content creator and skilled social media marketer for our company. 
                                In this dynamic role, your responsibility encompasses crafting top-notch content within the realm of topic, 
                                all while maintaining an ingenious, professional, and captivating tone. Your role includes creating a compelling content strategy, 
                                engaging with our audience, leveraging trends, analyzing insights, and staying at the forefront of industry trends to ensure our brand's online presence flourishes. 
                                Your content will not only resonate deeply with our target audience but also drive impactful results across diverse platforms.
                                So create content on this topic `{topic}` with `{tone}` tone and your goal is for target Audience .

                    {format_instructions}
                """

    # create prompt template

    prompt = ChatPromptTemplate(
        messages= HumanMessagePromptTemplate.from_template(template),
        input_variables=['topic', 'tone'],
        partial_variables={ 'format_instructions' : format_instructions},
        output_parser=output_parser
    )

    # create chain

    chain = LLMChain(llm=chat_llm, prompt=prompt)

    # final response 
    response = chain.predict_and_parse(topic=topic, tone=tone)

    return response



