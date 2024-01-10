import logging
from confluent_kafka import Consumer, Producer
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


category_schema = ResponseSchema(
    name = "category",
    description="Is the email an cancellation, review, or inquiry? Only provide these words.",
)

response_schemas = [category_schema]
parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()


template = """
Interpret the text and evaluate it.
category: Is the email an cancellation, review, or inquiry? Only provide these words.

Return only the JSON, do not add ANYTHING, NO INTERPRETATION!

text: {input}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=template)
chat = ChatOpenAI(temperature=0.0)

def delivery_report(err, msg):
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.info(f"Message delivered to {msg.topic()}")

def classify_email(email):
    messages = prompt.format_messages(
        input=email,
        format_instructions=format_instructions,
    )

    response = chat(messages)
    output_dict = parser.parse(response.content)

    return output_dict['category']

c = Consumer({
    'bootstrap.servers': 'kafka:29092',
    'group.id': 'email-classifier',
    'auto.offset.reset': 'earliest'
})

c.subscribe(['raw-emails'])

p = Producer({'bootstrap.servers': 'kafka:29092'})

while True:
    msg = c.poll(1.0)

    if msg is None:
        continue
    if msg.error():
        logger.error(f"Consumer error: {msg.error()}")
        continue

    email = msg.value().decode('utf-8')

    category = classify_email(email)

    logger.info(f"Categorized as {category}")

    logger.info(f"Classified email: {category}")

    topic_map = {
        "cancellation": "cancellation-emails",
        "review": "review-emails",
        "inquiry": "inquiry-emails",
    }

    p.produce(topic_map.get(category, 'unknown-category'), email, callback=delivery_report)
    p.flush()